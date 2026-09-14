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

-6.880749246503739

# 6. Current score

-10.31306

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65932) has done: 'I disable TensorFlow loading to avoid the protobuf import error and adjust the optimizer calls to use the current‑API argument `learning_rate` instead of the deprecated `lr`. This lets the fallback `DummyModel` run safely, fixes the training loop error, and ensures the script creates a proper `submission.csv` with the required columns.'
- What this solution (achieved -24.65932) has done: 'Implemented a lightweight fallback model that learns from the training features instead of returning a constant mean. Added `SimpleModel` (uses `GradientBoostingRegressor` to predict FVC and a calibrated confidence) and `SimpleQRModel` (outputs identical quantile predictions derived from the same regressor). Updated `QuantileRegressorMLP.get_model` to return these models when TensorFlow is unavailable, preserving the existing training loop and blending logic. This change keeps the overall architecture intact while providing more informative predictions, moving the score toward the target.'
- What this solution (achieved -24.65932) has done: 'I replace the final submission step with a model‑blending approach instead of using only the CV1 MLP predictions. Blending the three fold predictions (and optionally the QR predictions) typically yields a higher Laplace Log Likelihood, moving the score from –24.66 toward the target –6.88 while keeping the core pipeline unchanged.'
- What this solution (achieved -9.80652) has done: 'I fix the pipeline by storing each trained fallback model (MLP or QR) in the lists used later for inference, so the prediction step actually uses the learned models instead of zeros. This small change lets the blend step work with real predictions, moving the metric score closer to the target. No other logic is altered, preserving the original architecture and training flow.'
- What this solution (achieved -9.80713) has done: 'I add a lightweight feature‑scaling step to the fallback `SimpleModel` so that the GradientBoostingRegressor receives standardized inputs, which usually yields more accurate predictions and therefore a higher Laplace Log Likelihood. The change is confined to the model class, preserving the rest of the pipeline and submission format.'
- What this solution (achieved -10.93686) has done: 'I add a small helper in the `SubmissionPipeline` that evaluates the three CV‑based blends (each mixing the MLP and QR predictions) and automatically picks the one with the highest Laplace Log Likelihood score. Then the final submission use that best CV blend, which should move the score upward toward the target without altering the core modeling logic.'
- What this solution (achieved -10.31306) has done: 'I add a lightweight grid‑search inside the `best_cv_blend` method to find the mixing weight between the MLP and QR predictions that gives the highest Laplace Log‑Likelihood on the training OOF data for the selected CV. The chosen weight is then applied to the test predictions, keeping all existing logic unchanged while nudging the score upward toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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

TF_AVAILABLE = False

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        import tensorflow as tf

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
print(f'Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
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
        patient_dir = (
            f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}"
        )
        patient_directory = [
            pydicom.dcmread(os.path.join(patient_dir, s))
            for s in os.listdir(patient_dir)
        ]
        try:
            patient_directory.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            slice_positions = np.round(
                [s.ImagePositionPatient[2] for s in patient_directory], 4
            )
            non_duplicate_idx = np.unique(
                [
                    np.where(slice_position == slice_positions)[0][0]
                    for slice_position in slice_positions
                ]
            )
        except AttributeError:
            patient_directory.sort(key=lambda s: int(s.InstanceNumber))
            instance_numbers = np.array(
                [int(s.InstanceNumber) for s in patient_directory]
            )
            non_duplicate_idx = np.unique(
                [
                    np.where(instance_number == instance_numbers)[0][0]
                    for instance_number in instance_numbers
                ]
            )
        patient_directory = list(np.array(patient_directory)[non_duplicate_idx])

        metadata = {}
        pixel_spacings = np.zeros((len(patient_directory), 2))
        slice_positions = np.zeros((len(patient_directory)))

        for i, s in enumerate(patient_directory):
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
            (len(patient_directory), self.resize_shape[0], self.resize_shape[1]),
            dtype=np.int16,
        )
        for i, s in enumerate(patient_directory):
            s_cropped = self.crop_slice(s.pixel_array)
            s_resized = self.resize_slice(s_cropped)
            if not np.all(s_resized == 0):
                scan[i] = np.int16(s_resized)
        del patient_directory
        scan = scan[~np.all(scan == 0, axis=(-1, -2))]
        return scan, metadata

    def crop_slice(self, s):
        if np.all(s == 0):
            return s
        if s.shape[0] != self.resize_shape[0] or s.shape[1] != self.resize_shape[1]:
            s_cropped = s[~np.all(s == 0, axis=1)]
            s_cropped = s_cropped[:, ~np.all(s_cropped == 0, axis=0)]
        else:
            s_cropped = s
        return s_cropped

    def resize_slice(self, s):
        if s.shape[0] != self.resize_shape[0] or s.shape[1] != self.resize_shape[1]:
            s_resized = cv2.resize(
                s, self.resize_shape, interpolation=cv2.INTER_NEAREST
            )
        else:
            s_resized = s
        return s_resized

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
            fvc_val = (
                self.df_test[self.df_test["Patient"] == patient]["FVC"].values[0]
                if not self.df_test[self.df_test["Patient"] == patient]["FVC"].empty
                else np.nan
            )
            perc_val = (
                self.df_test[self.df_test["Patient"] == patient]["Percent"].values[0]
                if not self.df_test[self.df_test["Patient"] == patient]["Percent"].empty
                else np.nan
            )
            age_val = (
                self.df_test[self.df_test["Patient"] == patient]["Age"].values[0]
                if not self.df_test[self.df_test["Patient"] == patient]["Age"].empty
                else np.nan
            )
            sex_val = (
                self.df_test[self.df_test["Patient"] == patient]["Sex"].values[0]
                if not self.df_test[self.df_test["Patient"] == patient]["Sex"].empty
                else np.nan
            )
            smoke_val = (
                self.df_test[self.df_test["Patient"] == patient][
                    "SmokingStatus"
                ].values[0]
                if not self.df_test[self.df_test["Patient"] == patient][
                    "SmokingStatus"
                ].empty
                else np.nan
            )
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
        self.df_all["Age"] = self.df_all["Age"].astype(np.float32)
        self.df_all["FVC_Baseline"] = self.df_all["FVC_Baseline"].astype(np.float32)
        self.df_all["Percent"] = self.df_all["Percent"].astype(np.float32)
        self.df_all["Weeks_Passed"] = self.df_all["Weeks_Passed"].astype(np.float32)
        self.df_all["Weeks"] = self.df_all["Weeks"].astype(np.int16)
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
class DummyModel:
    """Simple fallback model used when TensorFlow is unavailable."""

    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.mean_fvc = 0.0
        self.confidence = 100.0  # constant confidence

    def fit(self, X, y, epochs=None, batch_size=None, verbose=0):
        if isinstance(y, (pd.Series, np.ndarray)):
            self.mean_fvc = float(np.mean(y))
        else:
            self.mean_fvc = float(np.mean(y.values))

    def predict(self, X):
        n = len(X)
        if self.output_dim == 2:
            return np.column_stack(
                [np.full(n, self.mean_fvc), np.full(n, self.confidence)]
            )
        elif self.output_dim == 3:
            return np.column_stack(
                [
                    np.full(n, self.mean_fvc),
                    np.full(n, self.mean_fvc),
                    np.full(n, self.mean_fvc),
                ]
            )
        else:
            return np.full((n, self.output_dim), self.mean_fvc)


class SimpleModel:
    """Lightweight learned fallback using GradientBoostingRegressor with feature scaling."""

    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.fvc_model = GradientBoostingRegressor(random_state=SEED)
        self.scaler = StandardScaler()
        self.confidence = 70.0  # will be calibrated after fitting

    def fit(self, X, y, epochs=None, batch_size=None, verbose=0):
        X_scaled = self.scaler.fit_transform(X)
        self.fvc_model.fit(X_scaled, y)
        residuals = y - self.fvc_model.predict(X_scaled)
        self.confidence = max(70.0, np.std(residuals))

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        n = X_scaled.shape[0]
        fvc_pred = self.fvc_model.predict(X_scaled)
        if self.output_dim == 2:
            return np.column_stack([fvc_pred, np.full(n, self.confidence)])
        else:
            return np.column_stack([fvc_pred, fvc_pred, fvc_pred])


class SimpleQRModel:
    """Fallback for the QR branch – uses the same regressor for all three quantiles."""

    def __init__(self):
        self.fvc_model = GradientBoostingRegressor(random_state=SEED)
        self.confidence = 70.0

    def fit(self, X, y, epochs=None, batch_size=None, verbose=0):
        self.fvc_model.fit(X, y)
        residuals = y - self.fvc_model.predict(X)
        self.confidence = max(70.0, np.std(residuals))

    def predict(self, X):
        n = X.shape[0]
        q_pred = self.fvc_model.predict(X)
        return np.column_stack([q_pred, q_pred, q_pred])


class QuantileRegressorMLP:

    def __init__(self, model, predictors, mlp_parameters, qr_parameters):
        self.model = model
        self.predictors = predictors
        self.mlp_parameters = mlp_parameters
        self.qr_parameters = qr_parameters

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return np.mean(score)

    def get_model(self, input_shape, m):
        if TF_AVAILABLE:
            pass
        else:
            if m == "MLP":
                return SimpleModel(output_dim=2)
            elif m == "QR":
                return SimpleQRModel()
            else:
                return DummyModel(output_dim=2)

    def train(self, X_train, y_train):
        self.mlp_scores_all = []
        self.qr_scores_all = []
        self.mlp_scores_last = []
        self.qr_scores_last = []

        self.mlp_oof = pd.DataFrame(np.zeros((len(y_train), 2)))
        self.qr_oof = pd.DataFrame(
            np.zeros((len(y_train), len(self.qr_parameters["quantiles"])))
        )

        self.mlp_models = {"CV1": [], "CV2": [], "CV3": []}
        self.qr_models = {"CV1": [], "CV2": [], "CV3": []}

        models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
        for m in models:
            print(f'\nRunning {m.upper()} Model\n{"-" * (14 + len(m))}')
            for cv in range(1, 4):
                for fold in sorted(X_train[f"CV{cv}_Fold"].unique()):
                    trn_idx = X_train[X_train[f"CV{cv}_Fold"] != fold].index
                    val_idx = X_train[X_train[f"CV{cv}_Fold"] == fold].index

                    X_trn = X_train.loc[trn_idx, self.predictors]
                    y_trn = y_train.loc[trn_idx]
                    X_val = X_train.loc[val_idx, self.predictors]
                    y_val = y_train.loc[val_idx]

                    model = self.get_model(input_shape=X_trn.shape[1], m=m)
                    if hasattr(model, "fit"):
                        if m == "MLP":
                            model.fit(
                                X_trn,
                                y_trn,
                                epochs=self.mlp_parameters["epochs"],
                                batch_size=self.mlp_parameters["batch_size"],
                                verbose=0,
                            )
                        else:  # QR
                            model.fit(
                                X_trn,
                                y_trn,
                                epochs=self.qr_parameters["epochs"],
                                batch_size=self.qr_parameters["batch_size"],
                                verbose=0,
                            )
                    else:
                        model.fit(X_trn, y_trn)

                    if m == "MLP":
                        self.mlp_models[f"CV{cv}"].append(model)
                    else:
                        self.qr_models[f"CV{cv}"].append(model)

                    predictions = model.predict(X_val)
                    if m == "MLP":
                        oof_predictions = predictions[:, 0]
                        self.mlp_oof.iloc[val_idx, 0] = oof_predictions
                        df_train.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = (
                            oof_predictions
                        )

                        oof_confidence = predictions[:, 1]
                        self.mlp_oof.iloc[val_idx, 1] = oof_confidence
                        df_train.loc[val_idx, f"CV{cv}_MLP_Confidence_Predictions"] = (
                            oof_confidence
                        )

                        fold_final_scores = []
                        for df_patient in np.array_split(
                            df_train.loc[val_idx]
                            .groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index(),
                            df_train.loc[val_idx, "Patient"].nunique(),
                        ):
                            fold_final_scores.append(
                                self.laplace_log_likelihood_metric(
                                    df_patient["FVC"],
                                    df_patient[f"CV{cv}_MLP_FVC_Predictions"],
                                    df_patient[f"CV{cv}_MLP_Confidence_Predictions"],
                                )
                            )

                    elif m == "QR":
                        oof_predictions = predictions[:, 1]
                        oof_confidence = predictions[:, 2] - predictions[:, 0]
                        for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                            self.qr_oof.iloc[val_idx, i] = predictions[:, i]
                            df_train.loc[
                                val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                            ] = predictions[:, i]

                        fold_final_scores = []
                        for df_patient in np.array_split(
                            df_train.loc[val_idx]
                            .groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index(),
                            df_train.loc[val_idx, "Patient"].nunique(),
                        ):
                            fold_final_scores.append(
                                self.laplace_log_likelihood_metric(
                                    df_patient["FVC"],
                                    df_patient[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                                    ],
                                    (
                                        df_patient[
                                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                                        ]
                                        - df_patient[
                                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                                        ]
                                    ),
                                )
                            )
                    oof_score_all = self.laplace_log_likelihood_metric(
                        y_val, oof_predictions, oof_confidence
                    )
                    print(
                        f"CV {cv} {m} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - All Measurement Score: {oof_score_all:.6} - Final 3 Measurement Score {np.mean(fold_final_scores):.6} [Std: {np.std(fold_final_scores):.6}]"
                    )

                oof_final_scores = []
                if m == "MLP":
                    for df_patient in np.array_split(
                        df_train.groupby("Patient").nth([-1, -2, -3]).reset_index(),
                        df_train["Patient"].nunique(),
                    ):
                        oof_final_scores.append(
                            self.laplace_log_likelihood_metric(
                                df_patient["FVC"],
                                df_patient[f"CV{cv}_MLP_FVC_Predictions"],
                                df_patient[f"CV{cv}_MLP_Confidence_Predictions"],
                            )
                        )
                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_train, self.mlp_oof.iloc[:, 0], self.mlp_oof.iloc[:, 1]
                    )
                else:  # QR
                    for df_patient in np.array_split(
                        df_train.groupby("Patient").nth([-1, -2, -3]).reset_index(),
                        df_train["Patient"].nunique(),
                    ):
                        oof_final_scores.append(
                            self.laplace_log_likelihood_metric(
                                df_patient["FVC"],
                                df_patient[
                                    f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                                ],
                                (
                                    df_patient[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                                    ]
                                    - df_patient[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                                    ]
                                ),
                            )
                        )
                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_train,
                        self.qr_oof.iloc[:, 1],
                        (self.qr_oof.iloc[:, 2] - self.qr_oof.iloc[:, 0]),
                    )
                print(
                    f'{"-" * 30}\\nCV {cv} {m} All Measurement OOF Score {oof_all_score:.6} - Final 3 Measurement OOF Score {np.mean(oof_final_scores):.6} [Std: {np.std(oof_final_scores):.6}]\\n{"-" * 30}\\n'
                )

    def predict(self, X_test):
        for cv in range(1, 4):
            mlp_predictions = np.zeros((len(X_test), 2))
            for model in self.mlp_models[f"CV{cv}"]:
                mlp_predictions += model.predict(X_test[self.predictors]) / len(
                    self.mlp_models[f"CV{cv}"]
                )
            X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_predictions[:, 0]
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_predictions[:, 1]

            qr_predictions = np.zeros(
                (len(X_test), len(self.qr_parameters["quantiles"]))
            )
            for model in self.qr_models[f"CV{cv}"]:
                qr_predictions += model.predict(X_test[self.predictors]) / len(
                    self.qr_models[f"CV{cv}"]
                )
            for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_predictions[:, i]

    def plot_predictions(self, df, patient):
        pass




## === cell 5
seed_everything(SEED)

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
    "mlp_parameters": {"lr": 0.0005, "epochs": 450, "batch_size": 2**5},
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




## === cell 6
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

    def single_model(self, model):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )
        prediction_cols = [
            col for col in self.df_train.columns if col.startswith(model)
        ]
        if model.split("_")[1] == "MLP":
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[prediction_cols[0]],
                self.df_train[prediction_cols[1]],
            )
            print(f"Single Model {model} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[prediction_cols[0]]
            self.df_test["Confidence"] = self.df_test[prediction_cols[1]]
        elif model.split("_")[1] == "QR":
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[prediction_cols[1]],
                (self.df_train[prediction_cols[2]] - self.df_train[prediction_cols[0]]),
            )
            print(f"Single Model {model} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[prediction_cols[1]]
            self.df_test["Confidence"] = (
                self.df_test[prediction_cols[2]] - self.df_test[prediction_cols[0]]
            )
        print(f'\n{self.df_test[["Patient_Week", "FVC", "Confidence"]].describe()}')
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)

    def blend(self, by, model, cv):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )
        if by == "model":
            if model == "MLP":
                for df in [self.df_train, self.df_test]:
                    df[f"{model}_FVC"] = (
                        df["CV1_MLP_FVC_Predictions"] * 0.34
                        + df["CV2_MLP_FVC_Predictions"] * 0.33
                        + df["CV3_MLP_FVC_Predictions"] * 0.33
                    )
                    df[f"{model}_Confidence"] = (
                        df["CV1_MLP_Confidence_Predictions"] * 0.34
                        + df["CV2_MLP_Confidence_Predictions"] * 0.33
                        + df["CV3_MLP_Confidence_Predictions"] * 0.33
                    )
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[f"{model}_FVC"],
                    self.df_train[f"{model}_Confidence"],
                )
                single_model_scores = [
                    round(
                        self.laplace_log_likelihood_metric(
                            self.df_train["FVC"],
                            self.df_train[f"CV{i}_MLP_FVC_Predictions"],
                            self.df_train[f"CV{i}_MLP_Confidence_Predictions"],
                        ),
                        6,
                    )
                    for i in range(1, 4)
                ]
                print(
                    f"MLP Blend Score: {score:.6} - Single Model Scores: {single_model_scores}"
                )
                self.df_test["FVC"] = self.df_test[f"{model}_FVC"]
                self.df_test["Confidence"] = self.df_test[f"{model}_Confidence"]
            elif model == "QR":
                quantiles = [0.25, 0.50, 0.75]
                for df in [self.df_train, self.df_test]:
                    for q in quantiles:
                        df[f"{model}_{q}_FVC"] = sum(
                            df[f"CV{i}_QR_{q}_Predictions"] * w
                            for i, w in enumerate([0.34, 0.33, 0.33], 1)
                        )
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[f"{model}_{quantiles[1]}_FVC"],
                    (
                        self.df_train[f"{model}_{quantiles[2]}_FVC"]
                        - self.df_train[f"{model}_{quantiles[0]}_FVC"]
                    ),
                )
                single_model_scores = [
                    round(
                        self.laplace_log_likelihood_metric(
                            self.df_train["FVC"],
                            self.df_train[f"CV{i}_QR_{quantiles[1]}_Predictions"],
                            (
                                self.df_train[f"CV{i}_QR_{quantiles[2]}_Predictions"]
                                - self.df_train[f"CV{i}_QR_{quantiles[0]}_Predictions"]
                            ),
                        ),
                        6,
                    )
                    for i in range(1, 4)
                ]
                print(
                    f"QR Blend Score: {score:.6} - Single Model Scores: {single_model_scores}"
                )
                self.df_test["FVC"] = self.df_test[f"{model}_{quantiles[1]}_FVC"]
                self.df_test["Confidence"] = (
                    self.df_test[f"{model}_{quantiles[2]}_FVC"]
                    - self.df_test[f"{model}_{quantiles[0]}_FVC"]
                )
        elif by == "cv":
            quantiles = [0.25, 0.50, 0.75]
            for df in [self.df_train, self.df_test]:
                df[f"CV{cv}_FVC"] = (
                    df[f"CV{cv}_MLP_FVC_Predictions"] * 0.5
                    + df[f"CV{cv}_QR_{quantiles[1]}_Predictions"] * 0.5
                )
                df[f"CV{cv}_Confidence"] = (
                    df[f"CV{cv}_MLP_Confidence_Predictions"] * 0.5
                    + (
                        df[f"CV{cv}_QR_{quantiles[2]}_Predictions"]
                        - df[f"CV{cv}_QR_{quantiles[0]}_Predictions"]
                    )
                    * 0.5
                )
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[f"CV{cv}_FVC"],
                self.df_train[f"CV{cv}_Confidence"],
            )
            single_model_scores = [
                round(
                    self.laplace_log_likelihood_metric(
                        self.df_train["FVC"],
                        self.df_train[f"CV{cv}_MLP_FVC_Predictions"],
                        self.df_train[f"CV{cv}_MLP_Confidence_Predictions"],
                    ),
                    6,
                ),
                round(
                    self.laplace_log_likelihood_metric(
                        self.df_train["FVC"],
                        self.df_train[f"CV{cv}_QR_{quantiles[1]}_Predictions"],
                        (
                            self.df_train[f"CV{cv}_QR_{quantiles[2]}_Predictions"]
                            - self.df_train[f"CV{cv}_QR_{quantiles[0]}_Predictions"]
                        ),
                    ),
                    6,
                ),
            ]
            print(
                f"CV{cv} Blend Score: {score:.6} - Single Model Scores: {single_model_scores}"
            )
            self.df_test["FVC"] = self.df_test[f"CV{cv}_FVC"]
            self.df_test["Confidence"] = self.df_test[f"CV{cv}_Confidence"]
        print(f'\n{self.df_test[["Patient_Week", "FVC", "Confidence"]].describe()}')
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)

    def best_cv_blend(self):
        """Select the best CV based on OOF score and also optimise the MLP‑QR mixing weight."""
        quantiles = [0.25, 0.50, 0.75]
        best_score = -np.inf
        best_cv = None
        best_weight = 0.5  # default equal weight

        for cv in [1, 2, 3]:
            fvc_eq = (
                self.df_train[f"CV{cv}_MLP_FVC_Predictions"] * 0.5
                + self.df_train[f"CV{cv}_QR_{quantiles[1]}_Predictions"] * 0.5
            )
            conf_eq = (
                self.df_train[f"CV{cv}_MLP_Confidence_Predictions"] * 0.5
                + (
                    self.df_train[f"CV{cv}_QR_{quantiles[2]}_Predictions"]
                    - self.df_train[f"CV{cv}_QR_{quantiles[0]}_Predictions"]
                )
                * 0.5
            )
            eq_score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"], fvc_eq, conf_eq
            )

            best_local_score = eq_score
            best_local_w = 0.5
            for w in np.arange(0.1, 1.0, 0.1):
                fvc_w = self.df_train[
                    f"CV{cv}_MLP_FVC_Predictions"
                ] * w + self.df_train[f"CV{cv}_QR_{quantiles[1]}_Predictions"] * (1 - w)
                conf_w = self.df_train[f"CV{cv}_MLP_Confidence_Predictions"] * w + (
                    self.df_train[f"CV{cv}_QR_{quantiles[2]}_Predictions"]
                    - self.df_train[f"CV{cv}_QR_{quantiles[0]}_Predictions"]
                ) * (1 - w)
                w_score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"], fvc_w, conf_w
                )
                if w_score > best_local_score:
                    best_local_score = w_score
                    best_local_w = w

            if best_local_score > best_score:
                best_score = best_local_score
                best_cv = cv
                best_weight = best_local_w

        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )
        w = best_weight
        self.df_test["FVC"] = self.df_test[
            f"CV{best_cv}_MLP_FVC_Predictions"
        ] * w + self.df_test[f"CV{best_cv}_QR_{quantiles[1]}_Predictions"] * (1 - w)
        self.df_test["Confidence"] = self.df_test[
            f"CV{best_cv}_MLP_Confidence_Predictions"
        ] * w + (
            self.df_test[f"CV{best_cv}_QR_{quantiles[2]}_Predictions"]
            - self.df_test[f"CV{best_cv}_QR_{quantiles[0]}_Predictions"]
        ) * (
            1 - w
        )
        print(
            f"Best CV blend chosen: CV{best_cv} with weight {best_weight:.2f} (MLP) giving score {best_score:.6}"
        )
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)


sub = SubmissionPipeline(df_train, df_test)
df_submission = sub.best_cv_blend()
df_submission.to_csv("submission.csv", index=False)

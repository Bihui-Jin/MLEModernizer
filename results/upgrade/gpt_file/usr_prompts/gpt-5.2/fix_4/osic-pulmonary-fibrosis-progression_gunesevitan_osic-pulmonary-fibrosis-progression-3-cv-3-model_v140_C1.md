# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-6.932888415138517

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pickle
import random
import gc

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from scipy.stats import skew, mode, kurtosis

import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import pydicom

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Lambda,
    Dropout,
    BatchNormalization,
    GaussianDropout,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import Callback

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
print(f'Test Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f"Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f"Sample Submission Shape = {df_submission.shape}")
print(
    f"Sample Submission Memory Usage = {df_submission.memory_usage().sum() / 1024 ** 2:.2f} MB"
)




## === cell 2
class TabularDataPreprocessor:

    def __init__(self, train, test, submission, n_folds, shuffle, ohe, scale):

        self.train = train.copy(deep=True)
        self.train.sort_values(by=["Patient", "Weeks"], inplace=True)
        self.test = test.copy(deep=True)
        self.submission = submission.copy(deep=True)

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
            df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1})
            df["Sex"] = df["Sex"].astype(np.uint8)
            df["SmokingStatus"] = df["SmokingStatus"].map(
                {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
            )
            df["SmokingStatus"] = df["SmokingStatus"].astype(np.uint8)

    def one_hot_encode(self):
        for df in [self.train, self.test]:
            df["Male"] = 0
            df["Female"] = 0
            df.loc[df["Sex"] == 0, "Male"] = 1
            df.loc[df["Sex"] == 1, "Female"] = 1

            df["Never smoked"] = 0
            df["Ex-smoker"] = 0
            df["Currently smokes"] = 0
            df.loc[df["SmokingStatus"] == 0, "Never smoked"] = 1
            df.loc[df["SmokingStatus"] == 1, "Ex-smoker"] = 1
            df.loc[df["SmokingStatus"] == 2, "Currently smokes"] = 1

            df.drop(columns=["Sex", "SmokingStatus"], inplace=True)

            for encoded_col in [
                "Male",
                "Female",
                "Never smoked",
                "Ex-smoker",
                "Currently smokes",
            ]:
                df[encoded_col] = df[encoded_col].astype(np.uint8)

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

        patients_unique = self.train["Patient"].unique()
        intercepts = np.empty(len(patients_unique), dtype=np.float32)
        coefs = np.empty(len(patients_unique), dtype=np.float32)

        for j, patient_name in enumerate(patients_unique):
            pat = self.train[self.train["Patient"] == patient_name]
            fvc_last2 = pat["FVC"].values[-2:]
            weeks_last2 = pat["Weeks"].values[-2:].reshape(-1, 1)

            std = fvc_last2.std()
            if std == 0 or np.isnan(std):
                z = np.zeros_like(fvc_last2, dtype=np.float32)
            else:
                z = (fvc_last2 - fvc_last2.mean()) / std

            reg = LinearRegression().fit(weeks_last2, z)
            intercepts[j] = reg.intercept_
            coefs[j] = reg.coef_[0]

        coef_map = dict(zip(patients_unique, coefs))
        int_map = dict(zip(patients_unique, intercepts))
        self.train["Intercept"] = self.train["Patient"].map(int_map).astype(np.float32)
        self.train["Coef"] = self.train["Patient"].map(coef_map).astype(np.float32)

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
        pw = (
            self.submission["Patient_Week"].astype(str).str.split("_", n=1, expand=True)
        )
        self.submission["Patient"] = pw[0].astype(str)
        self.submission["Weeks"] = pw[1].astype(int)
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

        df_all = pd.concat([self.train, self.test], ignore_index=True, axis=0)

        df_all["Age"] += df_all["Weeks_Passed"] / 52
        df_all["Age"] = df_all["Age"].astype(np.float32)
        df_all["FVC_Baseline"] = df_all["FVC_Baseline"].astype(np.float32)
        df_all["Percent"] = df_all["Percent"].astype(np.float32)
        df_all["Weeks_Passed"] = df_all["Weeks_Passed"].astype(np.float32)
        df_all["Weeks"] = df_all["Weeks"].astype(np.int16)

        df_all["FVC"] = df_all["FVC"].astype(np.float32)

        if self.scale:
            scale_features = ["FVC_Baseline", "Age", "Percent", "Weeks_Passed"]
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
    ohe=True,
    scale=False,
)

df_train, df_test = tabular_data_preprocessor.create_tabular_features()

print(
    f'Training Set (Tabular Features) Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(
    f"Training Set (Tabular Features) Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB"
)
print(
    f'Test Set (Tabular Features) Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}'
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

    def crop(self, s):
        if np.all(s == 0):
            return s

        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_cropped = s[~np.all(s == 0, axis=1)]
            s_cropped = s_cropped[:, ~np.all(s_cropped == 0, axis=0)]
        else:
            s_cropped = s

        return s_cropped

    def resize(self, s):
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_resized = cv2.resize(
                s, self.resize_shape, interpolation=cv2.INTER_NEAREST
            )
        else:
            s_resized = s

        return s_resized

    def window(self, s, slope, intercept, window_width, window_center, y_min, y_max):
        x = s * slope + intercept

        y = np.zeros_like(x)
        y[x <= (window_center - 0.5 - (window_width - 1) / 2)] = y_min
        y[x > (window_center - 0.5 + (window_width - 1) / 2)] = y_max
        mask = (x > (window_center - 0.5 - (window_width - 1) / 2)) & (
            x <= (window_center - 0.5 + (window_width - 1) / 2)
        )
        y[mask] = ((x[mask] - (window_center - 0.5)) / (window_width - 1) + 0.5) * (
            y_max - y_min
        ) + y_min

        return y

    def load_scan(self, dataset, patient_name):
        base_dir = (
            f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}"
        )

        files = os.listdir(base_dir)
        dsets = []
        for fn in files:
            fp = f"{base_dir}/{fn}"
            try:
                ds = pydicom.dcmread(
                    fp,
                    stop_before_pixels=False,
                    specific_tags=[
                        "PixelData",
                        "PixelSpacing",
                        "ImagePositionPatient",
                        "InstanceNumber",
                        "RescaleSlope",
                        "RescaleIntercept",
                    ],
                )
            except Exception:
                ds = pydicom.dcmread(fp)
            dsets.append(ds)
        patient_directory = dsets

        try:
            patient_directory.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            slice_positions = np.round(
                [float(s.ImagePositionPatient[2]) for s in patient_directory], 4
            )
            _, non_duplicate_idx = np.unique(slice_positions, return_index=True)
        except Exception:
            patient_directory.sort(key=lambda s: int(s.InstanceNumber))
            instance_numbers = np.array(
                [int(s.InstanceNumber) for s in patient_directory]
            )
            _, non_duplicate_idx = np.unique(instance_numbers, return_index=True)

        non_duplicate_idx = np.sort(non_duplicate_idx)
        patient_directory = [patient_directory[i] for i in non_duplicate_idx]

        metadata = {}
        pixel_spacings = np.zeros((len(patient_directory), 2))
        slice_positions = np.zeros((len(patient_directory)))

        for i, s in enumerate(patient_directory):
            try:
                pixel_spacings[i, :] = np.array(s.PixelSpacing, dtype=float)
            except Exception:
                pixel_spacings[i, :] = np.nan

            try:
                slice_positions[i] = float(s.ImagePositionPatient[2])
            except Exception:
                slice_positions[i] = np.nan

        metadata["PixelSpacing"] = list(np.round(np.nanmean(pixel_spacings, axis=0), 3))

        if patient_name == "ID00128637202219474716089":
            metadata["SliceSpacing"] = 5.0
        elif patient_name == "ID00132637202222178761324":
            metadata["SliceSpacing"] = 0.7
        else:
            diffs = np.abs(
                np.diff(np.round(slice_positions[~np.isnan(slice_positions)], 3))
            )
            if diffs.size == 0:
                metadata["SliceSpacing"] = 1.0
            else:
                try:
                    metadata["SliceSpacing"] = float(mode(diffs, keepdims=True).mode[0])
                except TypeError:
                    metadata["SliceSpacing"] = float(mode(diffs).mode[0])

        scan = np.zeros(
            (len(patient_directory), self.resize_shape[0], self.resize_shape[1]),
            dtype=np.int16,
        )
        for i, s in enumerate(patient_directory):
            s_processed = self.crop(s.pixel_array)
            s_processed = self.resize(s_processed)
            s_processed = self.window(
                s_processed,
                float(getattr(s, "RescaleSlope", 1.0)),
                float(getattr(s, "RescaleIntercept", 0.0)),
                self.window_width,
                self.window_center,
                self.y_min,
                self.y_max,
            )

            if np.all(s_processed == 0):
                continue
            else:
                scan[i] = np.int16(s_processed)

        del patient_directory
        scan = scan[~np.all(scan == 0, axis=(-1, -2))]
        return scan, metadata

    def _compute_patient_features(self, scan, metadata):
        volume = (
            (metadata["SliceSpacing"] * scan.shape[0])
            * (metadata["PixelSpacing"][0] * scan.shape[1])
            * (metadata["PixelSpacing"][1] * scan.shape[2])
        )
        voxel_vol = volume / (scan.shape[0] * scan.shape[1] * scan.shape[2])

        flat = scan.reshape(-1)
        f_mean = float(flat.mean())
        f_std = float(flat.std())
        f_var = float(flat.var())
        f_skew = float(skew(flat))
        f_kurt = float(kurtosis(flat))
        min_vol = float((scan == self.y_min).sum()) * float(voxel_vol)
        max_vol = float((scan == self.y_max).sum()) * float(voxel_vol)

        if scan.shape[0]:
            slice_skews = np.empty(scan.shape[0], dtype=np.float64)
            for i in range(scan.shape[0]):
                slice_skews[i] = skew(scan[i].reshape(-1))
            std_slice_skew = float(slice_skews.std())
            var_slice_skew = float(slice_skews.var())
        else:
            std_slice_skew = 0.0
            var_slice_skew = 0.0

        return {
            "VoxelVolume": float(voxel_vol),
            "Scan_Skew": f_skew,
            "Scan_Kurtosis": f_kurt,
            "Scan_Mean": f_mean,
            "Scan_Std": f_std,
            "Scan_Var": f_var,
            "Scan_Min_Volume": min_vol,
            "Scan_Max_Volume": max_vol,
            "Std_Slice_Skew": std_slice_skew,
            "Var_Slice_Skew": var_slice_skew,
        }

    def create_image_features(self):
        print(f'Creating Image Features for Training Set\n{"-" * 40}')

        cache_path = "df_scan_features_cache.pkl"
        if os.path.exists(cache_path):
            print(f"Loading cached image features: {cache_path}")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        else:
            cache = {}

        feat_path = (
            "../input/osic-pulmonary-fibrosis-progression-features/df_scan_features.csv"
        )
        if os.path.exists(feat_path):
            print(f"Image Features Imported: {feat_path}")
            df_train_features = pd.read_csv(feat_path)
            self.train = self.train.merge(df_train_features, on="Patient", how="left")
        else:
            print(
                "Precomputed features not found; computing from DICOMs (may take a few minutes)."
            )
            train_patients = self.train["Patient"].unique()
            for i, patient_name in enumerate(train_patients):
                if patient_name in cache:
                    continue
                try:
                    scan, metadata = self.load_scan("train", patient_name)
                except Exception as e:
                    print(f"[WARN] Failed reading train DICOMs for {patient_name}: {e}")
                    continue

                scan_size = scan.nbytes >> 20
                print(
                    f"[{i + 1}/{len(train_patients)}] {patient_name} Shape: {scan.shape} - Size: {scan_size} MB"
                )

                cache[patient_name] = self._compute_patient_features(scan, metadata)

                del scan, metadata
                gc.collect()

            with open(cache_path, "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

            df_feat = pd.DataFrame.from_dict(cache, orient="index").reset_index()
            df_feat.rename(columns={"index": "Patient"}, inplace=True)
            self.train = self.train.merge(df_feat, on="Patient", how="left")

        print(f'\nCreating Image Features for Test Set\n{"-" * 36}')
        test_patients = self.test["Patient"].unique()
        for i, patient_name in enumerate(test_patients):
            if patient_name in cache:
                for k, v in cache[patient_name].items():
                    self.test.loc[self.test["Patient"] == patient_name, k] = v
                continue

            try:
                scan, metadata = self.load_scan("test", patient_name)
            except Exception as e:
                print(f"[WARN] Failed reading test DICOMs for {patient_name}: {e}")
                continue

            scan_size = scan.nbytes >> 20
            print(
                f"[{i + 1}/{len(test_patients)}] {patient_name} Shape: {scan.shape} - Size: {scan_size} MB"
            )

            feats = self._compute_patient_features(scan, metadata)
            cache[patient_name] = feats
            for k, v in feats.items():
                self.test.loc[self.test["Patient"] == patient_name, k] = v

            del scan, metadata
            gc.collect()

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.scale:
            scale_features = [
                "Scan_Skew",
                "Scan_Kurtosis",
                "Scan_Mean",
                "Scan_Std",
                "Scan_Var",
                "Scan_Min_Volume",
                "Scan_Max_Volume",
            ]
            for col in scale_features:
                if col not in self.train.columns:
                    self.train[col] = np.nan
                if col not in self.test.columns:
                    self.test[col] = np.nan

            scaler = StandardScaler()
            scaler.fit(self.train.loc[:, scale_features].fillna(0.0))

            self.train.loc[:, scale_features] = scaler.transform(
                self.train.loc[:, scale_features].fillna(0.0)
            )
            self.test.loc[:, scale_features] = scaler.transform(
                self.test.loc[:, scale_features].fillna(0.0)
            )

        return self.train.copy(deep=True), self.test.drop(
            columns=["VoxelVolume"], errors="ignore"
        ).copy(deep=True)




## === cell 5
image_data_preprocessor = ImageDataPreprocessor(
    train=df_train,
    test=df_test,
    resize_shape=(512, 512),
    window_width=1500,
    window_center=-500,
    y_min=0,
    y_max=(2**8) - 1,
    scale=True,
)

df_train, df_test = image_data_preprocessor.create_image_features()

print(
    f'\nTraining Set (Image + Tabular Features) Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(
    f"Training Set (Image + Tabular Features) Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB"
)
print(
    f'Test Set (Image + Tabular Features) Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}'
)
print(
    f"Test Set (Image + Tabular Features) Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB"
)




## === cell 6
class QuantileRegressorMLP:

    def __init__(self, model, cv, predictors, mlp_parameters, qr_parameters):

        self.model = model
        self.cv = cv
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

    def laplace_log_likelihood_loss(self, y_true, y_pred):
        K.cast(y_true, "float32")
        K.cast(y_pred, "float32")

        sigma_lower_bound = K.constant(70, dtype="float32")
        delta_upper_bound = K.constant(1000, dtype="float32")

        sigma = y_pred[:, 1]
        fvc_pred = y_pred[:, 0]

        sigma_clipped = K.maximum(sigma, sigma_lower_bound)
        delta = K.abs(y_true[:, 0] - fvc_pred)
        delta_clipped = K.minimum(delta, delta_upper_bound)

        score = (delta_clipped / sigma_clipped) * K.sqrt(K.cast(2, "float32")) + K.log(
            sigma_clipped * K.sqrt(K.cast(2, "float32"))
        )
        return K.mean(score)

    def tilted_loss(self, y_true, y_pred):
        quantiles = K.constant(
            np.array([self.qr_parameters["quantiles"]]), dtype="float32"
        )
        error = y_true - y_pred
        return K.mean(K.maximum(quantiles * error, (quantiles - 1) * error))

    def hybrid_loss(self, w):
        def loss(y_true, y_pred):
            return w * self.tilted_loss(y_true, y_pred) + (
                1 - w
            ) * self.laplace_log_likelihood_loss(y_true, y_pred)

        return loss

    def get_model(self, input_shape, m):
        model = None

        if isinstance(input_shape, (int, np.integer)):
            input_shape = (int(input_shape),)
        else:
            input_shape = tuple(input_shape)

        if m == "MLP":
            input_layer = Input(shape=input_shape)
            x = Dense(2**7, activation="relu")(input_layer)
            x = GaussianDropout(0.01)(x)
            x = Dense(2**7, activation="relu")(x)
            x = GaussianDropout(0.01)(x)
            p1 = Dense(2, activation="linear")(x)
            p2 = Dense(2, activation="relu")(x)
            output_layer = Lambda(lambda x: x[0] + K.cumsum(x[1], axis=1))([p1, p2])

            model = Model(input_layer, output_layer)
            model.compile(
                loss=self.laplace_log_likelihood_loss,
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.mlp_parameters["lr"]
                ),
                metrics=[self.laplace_log_likelihood_loss],
            )

        elif m == "QR":
            input_layer = Input(shape=input_shape)
            x = Dense(2**7, activation="relu")(input_layer)
            x = GaussianDropout(0.01)(x)
            x = Dense(2**7, activation="relu")(input_layer if False else x)
            x = GaussianDropout(0.01)(x)
            p1 = Dense(3, activation="linear")(x)
            p2 = Dense(3, activation="relu")(x)
            output_layer = Lambda(lambda x: x[0] + K.cumsum(x[1], axis=1))([p1, p2])

            model = Model(input_layer, output_layer)
            model.compile(
                loss=self.tilted_loss,
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.qr_parameters["lr"]
                ),
                metrics=[self.laplace_log_likelihood_loss],
            )

        return model

    def _final3_patient_scores(self, df_ref, y_col, pred_col, sigma_col):
        last3 = df_ref.groupby("Patient").nth([-1, -2, -3])
        per_patient = last3.groupby(level=0).apply(
            lambda g: self.laplace_log_likelihood_metric(
                g[y_col].values, g[pred_col].values, g[sigma_col].values
            )
        )
        return per_patient.values

    def train(self, X_train, y_train, df_train_ref):
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

        X_mat = X_train[self.predictors].to_numpy(dtype=np.float32, copy=True)
        y_vec = y_train.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

        models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
        for m in models:
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + (len(m)))}')

            for cv in self.cv:
                folds = sorted(X_train[f"CV{cv}_Fold"].unique())
                for fold in folds:

                    trn_idx = X_train.index[X_train[f"CV{cv}_Fold"] != fold].to_numpy()
                    val_idx = X_train.index[X_train[f"CV{cv}_Fold"] == fold].to_numpy()

                    X_trn = X_mat[trn_idx]
                    y_trn = y_vec[trn_idx]
                    X_val = X_mat[val_idx]
                    y_val = y_vec[val_idx]

                    model = self.get_model(input_shape=X_trn.shape[1], m=m)
                    if m == "MLP":
                        model.fit(
                            X_trn,
                            y_trn,
                            epochs=self.mlp_parameters["epochs"],
                            batch_size=self.mlp_parameters["batch_size"],
                            verbose=0,
                        )
                        self.mlp_models[f"CV{cv}"].append(model)
                    elif m == "QR":
                        model.fit(
                            X_trn,
                            y_trn,
                            epochs=self.qr_parameters["epochs"],
                            batch_size=self.qr_parameters["batch_size"],
                            verbose=0,
                        )
                        self.qr_models[f"CV{cv}"].append(model)

                    predictions = model.predict(X_val, verbose=0)
                    if m == "MLP":
                        oof_predictions = predictions[:, 0]
                        self.mlp_oof.iloc[val_idx, 0] = oof_predictions
                        df_train_ref.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = (
                            oof_predictions
                        )

                        oof_confidence = predictions[:, 1]
                        self.mlp_oof.iloc[val_idx, 1] = oof_confidence
                        df_train_ref.loc[
                            val_idx, f"CV{cv}_MLP_Confidence_Predictions"
                        ] = oof_confidence

                        fold_final_scores = self._final3_patient_scores(
                            df_train_ref.loc[val_idx],
                            y_col="FVC",
                            pred_col=f"CV{cv}_MLP_FVC_Predictions",
                            sigma_col=f"CV{cv}_MLP_Confidence_Predictions",
                        )

                    elif m == "QR":
                        oof_predictions = predictions[:, 1]
                        oof_confidence = predictions[:, 2] - predictions[:, 0]
                        for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                            self.qr_oof.iloc[val_idx, i] = predictions[:, i]
                            df_train_ref.loc[
                                val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                            ] = predictions[:, i]

                        fold_final_scores = self._final3_patient_scores(
                            df_train_ref.loc[val_idx],
                            y_col="FVC",
                            pred_col=f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions',
                            sigma_col=None,
                        )
                        last3 = (
                            df_train_ref.loc[val_idx]
                            .groupby("Patient")
                            .nth([-1, -2, -3])
                        )
                        per_patient = last3.groupby(level=0).apply(
                            lambda g: self.laplace_log_likelihood_metric(
                                g["FVC"].values,
                                g[
                                    f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                                ].values,
                                (
                                    g[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                                    ].values
                                    - g[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                                    ].values
                                ),
                            )
                        )
                        fold_final_scores = per_patient.values

                    oof_score_all = self.laplace_log_likelihood_metric(
                        y_val.reshape(-1), oof_predictions, oof_confidence
                    )
                    print(
                        f"CV {cv} {m} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - All Measurement Score: {oof_score_all:.6} - Final 3 Measurement Score {np.mean(fold_final_scores):.6} [Std: {np.std(fold_final_scores):.6}]"
                    )

                if m == "MLP":
                    oof_final_scores = self._final3_patient_scores(
                        df_train_ref,
                        y_col="FVC",
                        pred_col=f"CV{cv}_MLP_FVC_Predictions",
                        sigma_col=f"CV{cv}_MLP_Confidence_Predictions",
                    )
                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_vec.reshape(-1),
                        self.mlp_oof.iloc[:, 0].values,
                        self.mlp_oof.iloc[:, 1].values,
                    )
                elif m == "QR":
                    last3 = df_train_ref.groupby("Patient").nth([-1, -2, -3])
                    per_patient = last3.groupby(level=0).apply(
                        lambda g: self.laplace_log_likelihood_metric(
                            g["FVC"].values,
                            g[
                                f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                            ].values,
                            (
                                g[
                                    f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                                ].values
                                - g[
                                    f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                                ].values
                            ),
                        )
                    )
                    oof_final_scores = per_patient.values
                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_vec.reshape(-1),
                        self.qr_oof.iloc[:, 1].values,
                        (self.qr_oof.iloc[:, 2].values - self.qr_oof.iloc[:, 0].values),
                    )

                print(
                    f'{"-" * 30}\nCV {cv} {m} All Measurement OOF Score {oof_all_score:.6} - Final 3 Measurement OOF Score {np.mean(oof_final_scores):.6} [Std: {np.std(oof_final_scores):.6}]\n{"-" * 30}\n'
                )

    def predict(self, X_test):
        X_mat = X_test[self.predictors].to_numpy(dtype=np.float32, copy=True)

        for cv in self.cv:
            mlp_predictions = np.zeros((len(X_test), 2), dtype=np.float32)
            for model in self.mlp_models[f"CV{cv}"]:
                mlp_predictions += model.predict(X_mat, verbose=0) / len(
                    self.mlp_models[f"CV{cv}"]
                )

            X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_predictions[:, 0]
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_predictions[:, 1]

            qr_predictions = np.zeros(
                (len(X_test), len(self.qr_parameters["quantiles"])), dtype=np.float32
            )
            for model in self.qr_models[f"CV{cv}"]:
                qr_predictions += model.predict(X_mat, verbose=0) / len(
                    self.qr_models[f"CV{cv}"]
                )

            for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_predictions[:, i]

    def plot_predictions(self, df, patient):
        if not hasattr(plt, "show"):
            return

        mlp_prediction_columns = [
            f"CV{cv}_MLP_{target}_Predictions"
            for cv in self.cv
            for target in ["FVC", "Confidence"]
        ]
        if "FVC" in df.columns and all(c in df.columns for c in mlp_prediction_columns):
            mlp_scores = []
            for cv in self.cv:
                score = self.laplace_log_likelihood_metric(
                    df["FVC"],
                    df[f"CV{cv}_MLP_FVC_Predictions"],
                    df[f"CV{cv}_MLP_Confidence_Predictions"],
                )
                mlp_scores.append(round(score, 5))
        else:
            mlp_scores = None

        qr_prediction_columns = [
            f"CV{cv}_QR_{quantile}_Predictions"
            for cv in self.cv
            for quantile in self.qr_parameters["quantiles"]
        ]
        if "FVC" in df.columns and all(c in df.columns for c in qr_prediction_columns):
            qr_scores = []
            for i, cv in enumerate(self.cv):
                score = self.laplace_log_likelihood_metric(
                    df["FVC"],
                    df[qr_prediction_columns[1 + (i * 3)]],
                    (
                        df[qr_prediction_columns[2 + (i * 3)]]
                        - df[qr_prediction_columns[0 + (i * 3)]]
                    ),
                )
                qr_scores.append(round(score, 5))
        else:
            qr_scores = None

        plot_cols = ["Weeks"]
        style = []
        if "FVC" in df.columns:
            plot_cols += ["FVC"]
            style += ["-b"]

        if all(c in df.columns for c in mlp_prediction_columns):
            plot_cols += mlp_prediction_columns[::2]
            style += ["r--", "g--", "b--"][: len(mlp_prediction_columns[::2])]

        if all(c in df.columns for c in qr_prediction_columns):
            plot_cols += qr_prediction_columns[1::3]
            style += ["r:", "g:", "b:"][: len(qr_prediction_columns[1::3])]

        if not all(col in df.columns for col in plot_cols):
            print(f"[WARN] Missing plot columns for patient {patient}; skipping plot.")
            return

        numeric_df = df[plot_cols].select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            print(
                f"[WARN] No numeric data to plot for patient {patient}; skipping plot."
            )
            return

        ax = df[plot_cols].set_index("Weeks").plot(figsize=(30, 6), style=style)

        if all(c in df.columns for c in qr_prediction_columns):
            ax.fill_between(
                df["Weeks"],
                df[qr_prediction_columns[2]],
                df[qr_prediction_columns[0]],
                alpha=0.1,
                label="CV1 QR Prediction Interval",
                color="red",
            )

        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if mlp_scores is not None and qr_scores is not None and "FVC" in df.columns:
            ax.set_title(
                f"Patient: {patient} - MLP Scores: {mlp_scores} QR Scores: {qr_scores}",
                size=25,
                pad=25,
            )
        else:
            ax.set_title(f"Patient: {patient}", size=25, pad=25)
        ax.legend(prop={"size": 18})

        plt.show()




## === cell 7
seed_everything(SEED)

X_train = df_train.drop(columns=["FVC", "Weeks"])
y_train = df_train["FVC"].copy(deep=True)

model_parameters = {
    "model": "Stack",
    "cv": [1],
    "predictors": [
        "Age",
        "Male",
        "Female",
        "Never smoked",
        "Ex-smoker",
        "Currently smokes",
        "FVC_Baseline",
        "Percent",
        "Weeks_Passed",
    ],
    "mlp_parameters": {"lr": 0.00025, "epochs": 800, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.00025,
        "epochs": 800,
        "batch_size": 2**5,
    },
}

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train, df_train_ref=df_train)
qr_mlp.predict(df_test)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: None

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3614915258.py in <cell line: 0>()
     28 
     29 qr_mlp = QuantileRegressorMLP(**model_parameters)
---> 30 qr_mlp.train(X_train, y_train, df_train_ref=df_train)
     31 qr_mlp.predict(df_test)
     32 

/tmp/ipykernel_11/3786321789.py in train(self, X_train, y_train, df_train_ref)
    198                             ] = predictions[:, i]
    199 
--> 200                         fold_final_scores = self._final3_patient_scores(
    201                             df_train_ref.loc[val_idx],
    202                             y_col="FVC",

/tmp/ipykernel_11/3786321789.py in _final3_patient_scores(self, df_ref, y_col, pred_col, sigma_col)
    105     def _final3_patient_scores(self, df_ref, y_col, pred_col, sigma_col):
    106         last3 = df_ref.groupby("Patient").nth([-1, -2, -3])
--> 107         per_patient = last3.groupby(level=0).apply(
    108             lambda g: self.laplace_log_likelihood_metric(
    109                 g[y_col].values, g[pred_col].values, g[sigma_col].values

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in apply(self, func, include_groups, *args, **kwargs)
   1822         with option_context("mode.chained_assignment", None):
   1823             try:
-> 1824                 result = self._python_apply_general(f, self._selected_obj)
   1825                 if (
   1826                     not isinstance(self.obj, Series)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _python_apply_general(self, f, data, not_indexed_same, is_transform, is_agg)
   1883             data after applying f
   1884         """
-> 1885         values, mutated = self._grouper.apply_groupwise(f, data, self.axis)
   1886         if not_indexed_same is None:
   1887             not_indexed_same = mutated

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in apply_groupwise(self, f, data, axis)
    917             # group might be modified
    918             group_axes = group.axes
--> 919             res = f(group)
    920             if not mutated and not _is_indexed_like(res, group_axes, axis):
    921                 mutated = True

/tmp/ipykernel_11/3786321789.py in <lambda>(g)
    107         per_patient = last3.groupby(level=0).apply(
    108             lambda g: self.laplace_log_likelihood_metric(
--> 109                 g[y_col].values, g[pred_col].values, g[sigma_col].values
    110             )
    111         )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: None

## === cell 8
if os.environ.get("RUN_PLOTS", "0") == "1":
    for patient, df in list(df_train.groupby("Patient"))[:3]:
        qr_mlp.plot_predictions(df, patient)




## === cell 9
if os.environ.get("RUN_PLOTS", "0") == "1":
    for patient, df in list(df_test.groupby("Patient"))[:2]:
        qr_mlp.plot_predictions(df, patient)




## === cell 10
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

        if model.split("_")[1] == "MLP":
            fvc_col = f"{model}_FVC_Predictions"
            conf_col = f"{model}_Confidence_Predictions"
            if (
                fvc_col not in self.df_train.columns
                or conf_col not in self.df_train.columns
            ):
                raise KeyError(
                    f"Missing required prediction columns: {fvc_col}, {conf_col}"
                )
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"], self.df_train[fvc_col], self.df_train[conf_col]
            )
            print(f"Single Model {model} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[fvc_col]
            self.df_test["Confidence"] = self.df_test[conf_col]

        elif model.split("_")[1] == "QR":
            quantiles = [0.25, 0.5, 0.75]
            q0 = f"{model}_{quantiles[0]}_Predictions"
            q1 = f"{model}_{quantiles[1]}_Predictions"
            q2 = f"{model}_{quantiles[2]}_Predictions"
            for c in [q0, q1, q2]:
                if c not in self.df_train.columns:
                    raise KeyError(f"Missing required prediction column: {c}")
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[q1],
                (self.df_train[q2] - self.df_train[q0]),
            )
            print(f"Single Model {model} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[q1]
            self.df_test["Confidence"] = self.df_test[q2] - self.df_test[q0]

        self.df_test["Confidence"] = (
            self.df_test["Confidence"].astype(np.float32).clip(lower=70)
        )

        print(f'\n{self.df_test[["Patient_Week", "FVC", "Confidence"]].describe()}')
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 11
sub = SubmissionPipeline(df_train, df_test)
df_submission = sub.single_model(model="CV1_MLP")
df_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_submission.shape)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'CV1_MLP_FVC_Predictions'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/29112149.py in <cell line: 0>()
      1 sub = SubmissionPipeline(df_train, df_test)
----> 2 df_submission = sub.single_model(model="CV1_MLP")
      3 df_submission.to_csv("submission.csv", index=False)
      4 print("Wrote submission.csv with shape:", df_submission.shape)
      5 

/tmp/ipykernel_11/331571884.py in single_model(self, model)
     34             )
     35             print(f"Single Model {model} Score: {score:.6}")
---> 36             self.df_test["FVC"] = self.df_test[fvc_col]
     37             self.df_test["Confidence"] = self.df_test[conf_col]
     38 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'CV1_MLP_FVC_Predictions'

## === cell 12
df_submission.head()

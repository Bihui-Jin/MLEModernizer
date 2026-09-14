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

-6.996737893464723

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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

import cv2
import pydicom

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

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

        for patient_name in self.train["Patient"].unique():
            last2 = self.train[(self.train["Patient"] == patient_name)]["FVC"].values[
                -2:
            ]
            std = last2.std()
            if std == 0 or np.isnan(std):
                z = np.zeros_like(last2, dtype=np.float32)
            else:
                z = (last2 - last2.mean()) / std

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
            (self.train["Coef"] <= 0.4) & (self.train["Coef"] >= -0.4), "Cluster"
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
        self.submission["Patient"] = (
            self.submission["Patient_Week"].apply(lambda x: x.split("_")[0]).astype(str)
        )
        self.submission["Weeks"] = (
            self.submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
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

        df_all = pd.concat([self.train, self.test], ignore_index=True, axis=0)

        df_all["Age"] += df_all["Weeks_Passed"] / 52
        df_all["Age"] = df_all["Age"].astype(np.float32)
        df_all["FVC_Baseline"] = df_all["FVC_Baseline"].astype(np.float32)
        df_all["Percent"] = df_all["Percent"].astype(np.float32)
        df_all["Weeks_Passed"] = df_all["Weeks_Passed"].astype(np.float32)
        df_all["Weeks"] = df_all["Weeks"].astype(np.int16)
        df_all["FVC"] = df_all["FVC"].astype(np.float32)

        if self.scale:
            scale_features = ["Age", "Percent", "Weeks_Passed", "FVC_Baseline"]
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
    scale=True,
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
        m = (x > (window_center - 0.5 - (window_width - 1) / 2)) & (
            x <= (window_center - 0.5 + (window_width - 1) / 2)
        )
        y[m] = ((x[m] - (window_center - 0.5)) / (window_width - 1) + 0.5) * (
            y_max - y_min
        ) + y_min
        return y

    def load_scan(self, dataset, patient_name):
        patient_directory = [
            pydicom.dcmread(
                f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}/{s}"
            )
            for s in os.listdir(
                f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}"
            )
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
        except Exception:
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
            except Exception:
                pixel_spacings[i, :] = np.nan

            try:
                slice_positions[i] = s.ImagePositionPatient[2]
            except Exception:
                continue

        metadata["PixelSpacing"] = list(np.round(np.nanmean(pixel_spacings, axis=0), 3))

        if patient_name == "ID00128637202219474716089":
            metadata["SliceSpacing"] = 5.0
        elif patient_name == "ID00132637202222178761324":
            metadata["SliceSpacing"] = 0.7
        else:
            diffs = np.abs(np.diff(np.round(slice_positions, 3)))
            diffs = diffs[diffs > 0]
            if len(diffs) == 0:
                metadata["SliceSpacing"] = 1.0
            else:
                metadata["SliceSpacing"] = mode(diffs, keepdims=True).mode[0]

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
            scan[i] = np.int16(s_processed)

        del patient_directory
        scan = scan[~np.all(scan == 0, axis=(-1, -2))]
        return scan, metadata

    def create_image_features(self):
        print(f'Creating Image Features for Training Set\n{"-" * 40}')
        for i, patient_name in enumerate(self.train["Patient"].unique()):
            scan, metadata = self.load_scan("train", patient_name)
            scan_size = scan.nbytes >> 20
            print(
                f'[{i + 1}/{len(self.train["Patient"].unique())}] {patient_name} Shape: {scan.shape} - Size: {scan_size} MB'
            )

            volume = (
                (metadata["SliceSpacing"] * scan.shape[0])
                * (metadata["PixelSpacing"][0] * scan.shape[1])
                * (metadata["PixelSpacing"][1] * scan.shape[2])
            )
            self.train.loc[self.train["Patient"] == patient_name, "VoxelVolume"] = (
                volume / (scan.shape[0] * scan.shape[1] * scan.shape[2])
            )

            flat = scan.flatten()
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Skew"] = skew(
                flat
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Kurtosis"] = (
                kurtosis(flat)
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Mean"] = (
                flat.mean()
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Std"] = (
                flat.std()
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Var"] = (
                flat.var()
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Min_Volume"] = (
                scan == self.y_min
            ).sum() * float(
                self.train.loc[
                    self.train["Patient"] == patient_name, "VoxelVolume"
                ].iloc[0]
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Max_Volume"] = (
                scan == self.y_max
            ).sum() * float(
                self.train.loc[
                    self.train["Patient"] == patient_name, "VoxelVolume"
                ].iloc[0]
            )

            slice_skews = (
                [skew(s.flatten()) for s in scan] if scan.shape[0] > 0 else [0.0]
            )
            self.train.loc[self.train["Patient"] == patient_name, "Std_Slice_Skew"] = (
                np.std(slice_skews)
            )
            self.train.loc[self.train["Patient"] == patient_name, "Var_Slice_Skew"] = (
                np.var(slice_skews)
            )

            del scan, metadata, volume, flat, slice_skews
            gc.collect()

        print(f'\nCreating Image Features for Test Set\n{"-" * 36}')
        for i, patient_name in enumerate(self.test["Patient"].unique()):
            scan, metadata = self.load_scan("test", patient_name)
            scan_size = scan.nbytes >> 20
            print(
                f'[{i + 1}/{len(self.test["Patient"].unique())}] {patient_name} Shape: {scan.shape} - Size: {scan_size} MB'
            )

            volume = (
                (metadata["SliceSpacing"] * scan.shape[0])
                * (metadata["PixelSpacing"][0] * scan.shape[1])
                * (metadata["PixelSpacing"][1] * scan.shape[2])
            )
            self.test.loc[self.test["Patient"] == patient_name, "VoxelVolume"] = (
                volume / (scan.shape[0] * scan.shape[1] * scan.shape[2])
            )

            flat = scan.flatten()
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Skew"] = skew(
                flat
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Kurtosis"] = (
                kurtosis(flat)
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Mean"] = (
                flat.mean()
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Std"] = flat.std()
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Var"] = flat.var()
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Min_Volume"] = (
                scan == self.y_min
            ).sum() * float(
                self.test.loc[self.test["Patient"] == patient_name, "VoxelVolume"].iloc[
                    0
                ]
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Max_Volume"] = (
                scan == self.y_max
            ).sum() * float(
                self.test.loc[self.test["Patient"] == patient_name, "VoxelVolume"].iloc[
                    0
                ]
            )

            slice_skews = (
                [skew(s.flatten()) for s in scan] if scan.shape[0] > 0 else [0.0]
            )
            self.test.loc[self.test["Patient"] == patient_name, "Std_Slice_Skew"] = (
                np.std(slice_skews)
            )
            self.test.loc[self.test["Patient"] == patient_name, "Var_Slice_Skew"] = (
                np.var(slice_skews)
            )

            del scan, metadata, volume, flat, slice_skews
            gc.collect()

        if self.scale:
            scale_features = [
                "Scan_Skew",
                "Scan_Kurtosis",
                "Scan_Mean",
                "Scan_Std",
                "Scan_Var",
                "Scan_Min_Volume",
                "Scan_Max_Volume",
                "Std_Slice_Skew",
                "Var_Slice_Skew",
            ]
            scaler = StandardScaler()
            scaler.fit(self.train.loc[:, scale_features])

            self.train.loc[:, scale_features] = scaler.transform(
                self.train.loc[:, scale_features]
            )
            self.test.loc[:, scale_features] = scaler.transform(
                self.test.loc[:, scale_features]
            )

        return self.train.copy(deep=True), self.test.drop(columns=["VoxelVolume"]).copy(
            deep=True
        )




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




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1439603780.py in <cell line: 0>()
     11 )
     12 
---> 13 df_train, df_test = image_data_preprocessor.create_image_features()
     14 
     15 print(

/tmp/ipykernel_11/2893199340.py in create_image_features(self)
    149         print(f'Creating Image Features for Training Set\n{"-" * 40}')
    150         for i, patient_name in enumerate(self.train["Patient"].unique()):
--> 151             scan, metadata = self.load_scan("train", patient_name)
    152             scan_size = scan.nbytes >> 20
    153             print(

/tmp/ipykernel_11/2893199340.py in load_scan(self, dataset, patient_name)
    126         )
    127         for i, s in enumerate(patient_directory):
--> 128             s_processed = self.crop(s.pixel_array)
    129             s_processed = self.resize(s_processed)
    130             s_processed = self.window(

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    980             cast(
    981                 dict[str, "DecodeFunction"],
--> 982                 self._validate_plugins(decoding_plugin),
    983             ),
    984         )

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_plugins(self, plugin)
    255         missing = "\n".join([f"\t{s}" for s in self.missing_dependencies])
    256         if self._decoder:
--> 257             raise RuntimeError(
    258                 f"Unable to decompress '{self.UID.name}' pixel data because all "
    259                 f"plugins are missing dependencies:\n{missing}"

RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1

## === cell 6
class QuantileRegressorMLP:
    """
    TensorFlow-free drop-in replacement:
    - MLP model outputs [FVC, Confidence]
    - QR model outputs 3 quantiles [q0.2, q0.5, q0.8]
    Uses sklearn MLPRegressor but keeps same training loops, folds, and column outputs.
    """

    def __init__(self, model, cv, predictors, mlp_parameters, qr_parameters):
        self.model = model
        self.cv = cv
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
        return float(np.mean(score))

    def _fit_mlp(self, X_trn, y_trn):
        model = MLPRegressor(
            hidden_layer_sizes=(2**7, 2**7),
            activation="relu",
            solver="adam",
            learning_rate_init=self.mlp_parameters["lr"],
            max_iter=self.mlp_parameters["epochs"],
            batch_size=self.mlp_parameters["batch_size"],
            random_state=SEED,
            shuffle=True,
            verbose=False,
        )
        model.fit(X_trn, y_trn)
        return model

    def _fit_qr(self, X_trn, y_trn, quantile):
        y = y_trn.reshape(-1)
        med = np.median(y)
        w = np.where(y >= med, quantile, 1.0 - quantile).astype(np.float32)
        model = MLPRegressor(
            hidden_layer_sizes=(2**7, 2**7),
            activation="relu",
            solver="adam",
            learning_rate_init=self.qr_parameters["lr"],
            max_iter=self.qr_parameters["epochs"],
            batch_size=self.qr_parameters["batch_size"],
            random_state=SEED,
            shuffle=True,
            verbose=False,
        )
        model.fit(X_trn, y, sample_weight=w)
        return model

    def train(self, X_train, y_train):
        global df_train  # preserves original side-effect usage

        self.mlp_oof = pd.DataFrame(
            np.zeros((len(y_train), 2)), columns=["FVC", "Conf"]
        )
        self.qr_oof = pd.DataFrame(
            np.zeros((len(y_train), len(self.qr_parameters["quantiles"]))),
            columns=[str(q) for q in self.qr_parameters["quantiles"]],
        )

        self.mlp_models = {f"CV{i}": [] for i in [1, 2, 3]}
        self.qr_models = {f"CV{i}": [] for i in [1, 2, 3]}

        models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
        for m in models:
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + (len(m)))}')

            for cv in self.cv:
                fold_col = f"CV{cv}_Fold"
                if fold_col not in X_train.columns:
                    raise KeyError(
                        f"Missing {fold_col} in X_train. Available columns: {list(X_train.columns)}"
                    )

                for fold in sorted(X_train[fold_col].unique()):
                    trn_idx = X_train.loc[X_train[fold_col] != fold].index
                    val_idx = X_train.loc[X_train[fold_col] == fold].index

                    X_trn = X_train.loc[trn_idx, self.predictors].to_numpy(
                        dtype=np.float32
                    )
                    y_trn = y_train.loc[trn_idx].to_numpy(dtype=np.float32).reshape(-1)
                    X_val = X_train.loc[val_idx, self.predictors].to_numpy(
                        dtype=np.float32
                    )
                    y_val = y_train.loc[val_idx].to_numpy(dtype=np.float32).reshape(-1)

                    if m == "MLP":
                        model_fvc = self._fit_mlp(X_trn, y_trn)
                        resid_trn = y_trn - model_fvc.predict(X_trn)
                        model_sigma = self._fit_mlp(X_trn, np.abs(resid_trn))
                        self.mlp_models[f"CV{cv}"].append((model_fvc, model_sigma))

                        pred_fvc = model_fvc.predict(X_val)
                        pred_sigma = np.maximum(model_sigma.predict(X_val), 1.0)

                        self.mlp_oof.iloc[val_idx, 0] = pred_fvc
                        self.mlp_oof.iloc[val_idx, 1] = pred_sigma
                        df_train.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = pred_fvc
                        df_train.loc[val_idx, f"CV{cv}_MLP_Confidence_Predictions"] = (
                            pred_sigma
                        )

                        fold_final_scores = []
                        val_patients = (
                            df_train.loc[val_idx]
                            .groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index()
                        )
                        for _, df_patient in val_patients.groupby("Patient"):
                            fold_final_scores.append(
                                self.laplace_log_likelihood_metric(
                                    df_patient["FVC"].values,
                                    df_patient[f"CV{cv}_MLP_FVC_Predictions"].values,
                                    df_patient[
                                        f"CV{cv}_MLP_Confidence_Predictions"
                                    ].values,
                                )
                            )

                        oof_score_all = self.laplace_log_likelihood_metric(
                            y_val, pred_fvc, pred_sigma
                        )
                        print(
                            f"CV {cv} {m} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - All Measurement Score: {oof_score_all:.6} - Final 3 Measurement Score {np.mean(fold_final_scores):.6} [Std: {np.std(fold_final_scores):.6}]"
                        )

                    elif m == "QR":
                        q_models = []
                        q_preds = []
                        for quantile in self.qr_parameters["quantiles"]:
                            qm = self._fit_qr(X_trn, y_trn, quantile=quantile)
                            q_models.append(qm)
                            q_preds.append(qm.predict(X_val))
                        q_preds = np.vstack(q_preds).T
                        q_preds.sort(axis=1)

                        self.qr_models[f"CV{cv}"].append(q_models)

                        for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                            self.qr_oof.iloc[val_idx, i] = q_preds[:, i]
                            df_train.loc[
                                val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                            ] = q_preds[:, i]

                        oof_predictions = q_preds[:, 1]
                        oof_confidence = q_preds[:, 2] - q_preds[:, 0]

                        fold_final_scores = []
                        val_patients = (
                            df_train.loc[val_idx]
                            .groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index()
                        )
                        for _, df_patient in val_patients.groupby("Patient"):
                            fold_final_scores.append(
                                self.laplace_log_likelihood_metric(
                                    df_patient["FVC"].values,
                                    df_patient[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                                    ].values,
                                    (
                                        df_patient[
                                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                                        ].values
                                        - df_patient[
                                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                                        ].values
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
                    pats = df_train.groupby("Patient").nth([-1, -2, -3]).reset_index()
                    for _, df_patient in pats.groupby("Patient"):
                        oof_final_scores.append(
                            self.laplace_log_likelihood_metric(
                                df_patient["FVC"].values,
                                df_patient[f"CV{cv}_MLP_FVC_Predictions"].values,
                                df_patient[f"CV{cv}_MLP_Confidence_Predictions"].values,
                            )
                        )
                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_train.values,
                        self.mlp_oof.iloc[:, 0].values,
                        self.mlp_oof.iloc[:, 1].values,
                    )

                elif m == "QR":
                    pats = df_train.groupby("Patient").nth([-1, -2, -3]).reset_index()
                    for _, df_patient in pats.groupby("Patient"):
                        oof_final_scores.append(
                            self.laplace_log_likelihood_metric(
                                df_patient["FVC"].values,
                                df_patient[
                                    f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                                ].values,
                                (
                                    df_patient[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                                    ].values
                                    - df_patient[
                                        f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                                    ].values
                                ),
                            )
                        )
                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_train.values,
                        self.qr_oof.iloc[:, 1].values,
                        (self.qr_oof.iloc[:, 2].values - self.qr_oof.iloc[:, 0].values),
                    )

                print(
                    f'{"-" * 30}\nCV {cv} {m} All Measurement OOF Score {oof_all_score:.6} - Final 3 Measurement OOF Score {np.mean(oof_final_scores):.6} [Std: {np.std(oof_final_scores):.6}]\n{"-" * 30}\n'
                )

    def predict(self, X_test):
        for cv in self.cv:
            mlp_pred_fvc = np.zeros(len(X_test), dtype=np.float32)
            mlp_pred_sig = np.zeros(len(X_test), dtype=np.float32)
            for model_fvc, model_sigma in self.mlp_models[f"CV{cv}"]:
                X = X_test[self.predictors].to_numpy(dtype=np.float32)
                mlp_pred_fvc += model_fvc.predict(X) / len(self.mlp_models[f"CV{cv}"])
                mlp_pred_sig += np.maximum(model_sigma.predict(X), 1.0) / len(
                    self.mlp_models[f"CV{cv}"]
                )

            X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_pred_fvc
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_pred_sig

            qr_preds = np.zeros(
                (len(X_test), len(self.qr_parameters["quantiles"])), dtype=np.float32
            )
            for q_models in self.qr_models[f"CV{cv}"]:
                X = X_test[self.predictors].to_numpy(dtype=np.float32)
                fold_preds = np.vstack([m.predict(X) for m in q_models]).T
                fold_preds.sort(axis=1)
                qr_preds += fold_preds / len(self.qr_models[f"CV{cv}"])

            qr_preds.sort(axis=1)
            for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_preds[:, i]

    def plot_predictions(self, df, patient):
        mlp_prediction_columns = [
            f"CV{cv}_MLP_{target}_Predictions"
            for cv in self.cv
            for target in ["FVC", "Confidence"]
        ]
        qr_prediction_columns = [
            f"CV{cv}_QR_{quantile}_Predictions"
            for cv in self.cv
            for quantile in self.qr_parameters["quantiles"]
        ]

        cols = (
            ["Weeks"]
            + (["FVC"] if "FVC" in df.columns else [])
            + mlp_prediction_columns[::2]
            + qr_prediction_columns[1::3]
        )
        ax = df[cols].set_index("Weeks").plot(figsize=(30, 6))
        ax.set_title(f"Patient: {patient}", size=25, pad=25)
        plt.show()




## === cell 7
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
        "Scan_Skew",
        "Scan_Kurtosis",
        "Scan_Mean",
        "Scan_Std",
        "Scan_Var",
        "Scan_Min_Volume",
        "Scan_Max_Volume",
        "Std_Slice_Skew",
        "Var_Slice_Skew",
    ],
    "mlp_parameters": {"lr": 0.0005, "epochs": 300, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.2, 0.5, 0.8],
        "lr": 0.0005,
        "epochs": 300,
        "batch_size": 2**5,
    },
}

missing = [c for c in model_parameters["predictors"] if c not in df_train.columns]
if missing:
    raise KeyError(f"Missing predictors in df_train: {missing}")

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train)
qr_mlp.predict(df_test)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1940554754.py in <cell line: 0>()
     38 missing = [c for c in model_parameters["predictors"] if c not in df_train.columns]
     39 if missing:
---> 40     raise KeyError(f"Missing predictors in df_train: {missing}")
     41 
     42 qr_mlp = QuantileRegressorMLP(**model_parameters)

KeyError: "Missing predictors in df_train: ['Scan_Skew', 'Scan_Kurtosis', 'Scan_Mean', 'Scan_Std', 'Scan_Var', 'Scan_Min_Volume', 'Scan_Max_Volume', 'Std_Slice_Skew', 'Var_Slice_Skew']"

## === cell 8
for patient, dfp in list(df_train.groupby("Patient"))[:2]:
    qr_mlp.plot_predictions(dfp, patient)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4241823846.py in <cell line: 0>()
      1 # Optional sanity plots (kept but limited to avoid heavy output)
      2 for patient, dfp in list(df_train.groupby("Patient"))[:2]:
----> 3     qr_mlp.plot_predictions(dfp, patient)
      4 
      5 

NameError: name 'qr_mlp' is not defined

## === cell 9
class SubmissionPipeline:

    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(np.asarray(sigma).reshape(-1), 70)
        delta_clipped = np.minimum(
            np.abs(np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)),
            1000,
        )
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return float(np.mean(score))

    def blend(self, by, model, cv):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )

        if by == "cv":
            quantiles = [0.2, 0.5, 0.8]
            for df in [self.df_train, self.df_test]:
                df[f"CV{cv}_FVC"] = (df[f"CV{cv}_MLP_FVC_Predictions"] * 0.5) + (
                    df[f"CV{cv}_QR_{quantiles[1]}_Predictions"] * 0.5
                )
                df[f"CV{cv}_Confidence"] = (
                    df[f"CV{cv}_MLP_Confidence_Predictions"] * 0.5
                ) + (
                    (
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
            print(f"CV{cv} Blend OOF Score: {score:.6}")

            self.df_test["FVC"] = self.df_test[f"CV{cv}_FVC"]
            self.df_test["Confidence"] = self.df_test[f"CV{cv}_Confidence"]

        self.df_test["Confidence"] = np.maximum(
            self.df_test["Confidence"].astype(np.float32), 70.0
        )

        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 10
sub = SubmissionPipeline(df_train, df_test)
df_sub = sub.blend(by="cv", model=None, cv=1)

sample = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
df_sub = sample[["Patient_Week"]].merge(df_sub, on="Patient_Week", how="left")

df_sub["FVC"] = df_sub["FVC"].fillna(sample["FVC"]).astype(np.float32)
df_sub["Confidence"] = (
    df_sub["Confidence"].fillna(sample["Confidence"]).astype(np.float32)
)

df_sub.to_csv("submission.csv", index=False)
print(df_sub.head())
print("\nWrote submission.csv with shape:", df_sub.shape)



## --- ERROR in cell 10, traceback:
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
/tmp/ipykernel_11/446777826.py in <cell line: 0>()
      1 sub = SubmissionPipeline(df_train, df_test)
----> 2 df_sub = sub.blend(by="cv", model=None, cv=1)
      3 
      4 # Ensure submission matches sample ordering exactly (safe alignment)
      5 sample = pd.read_csv(

/tmp/ipykernel_11/3106431636.py in blend(self, by, model, cv)
     26             quantiles = [0.2, 0.5, 0.8]
     27             for df in [self.df_train, self.df_test]:
---> 28                 df[f"CV{cv}_FVC"] = (df[f"CV{cv}_MLP_FVC_Predictions"] * 0.5) + (
     29                     df[f"CV{cv}_QR_{quantiles[1]}_Predictions"] * 0.5
     30                 )

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

## === cell 11
df_sub

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/944308094.py in <cell line: 0>()
----> 1 df_sub

NameError: name 'df_sub' is not defined

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

-6.913975360666762

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

import cv2
import pydicom

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, QuantileRegressor
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
                z = np.zeros_like(last2, dtype=float)
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
            md = mode(np.abs(np.diff(np.round(slice_positions, 3))), keepdims=True)
            metadata["SliceSpacing"] = (
                float(md.mode[0]) if hasattr(md, "mode") else float(list(md)[0][0])
            )

        scan = np.zeros(
            (len(patient_directory), self.resize_shape[0], self.resize_shape[1]),
            dtype=np.int16,
        )
        for i, s in enumerate(patient_directory):
            s_processed = self.crop(s.pixel_array)
            s_processed = self.resize(s_processed)
            s_processed = self.window(
                s_processed,
                s.RescaleSlope,
                s.RescaleIntercept,
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

            self.train.loc[self.train["Patient"] == patient_name, "Scan_Skew"] = skew(
                scan.flatten()
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Kurtosis"] = (
                kurtosis(scan.flatten())
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Mean"] = (
                scan.flatten().mean()
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Std"] = (
                scan.flatten().std()
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Var"] = (
                scan.flatten().var()
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Min_Volume"] = (
                scan[scan == self.y_min].shape[0]
                * self.train.loc[self.train["Patient"] == patient_name, "VoxelVolume"]
            )
            self.train.loc[self.train["Patient"] == patient_name, "Scan_Max_Volume"] = (
                scan[scan == self.y_max].shape[0]
                * self.train.loc[self.train["Patient"] == patient_name, "VoxelVolume"]
            )

            slice_skews = [skew(s.flatten()) for s in scan] if len(scan) else [0.0]
            self.train.loc[self.train["Patient"] == patient_name, "Std_Slice_Skew"] = (
                np.std(slice_skews)
            )
            self.train.loc[self.train["Patient"] == patient_name, "Var_Slice_Skew"] = (
                np.var(slice_skews)
            )

            del scan, metadata, volume, slice_skews
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

            self.test.loc[self.test["Patient"] == patient_name, "Scan_Skew"] = skew(
                scan.flatten()
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Kurtosis"] = (
                kurtosis(scan.flatten())
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Mean"] = (
                scan.flatten().mean()
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Std"] = (
                scan.flatten().std()
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Var"] = (
                scan.flatten().var()
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Min_Volume"] = (
                scan[scan == self.y_min].shape[0]
                * self.test.loc[self.test["Patient"] == patient_name, "VoxelVolume"]
            )
            self.test.loc[self.test["Patient"] == patient_name, "Scan_Max_Volume"] = (
                scan[scan == self.y_max].shape[0]
                * self.test.loc[self.test["Patient"] == patient_name, "VoxelVolume"]
            )

            slice_skews = [skew(s.flatten()) for s in scan] if len(scan) else [0.0]
            self.test.loc[self.test["Patient"] == patient_name, "Std_Slice_Skew"] = (
                np.std(slice_skews)
            )
            self.test.loc[self.test["Patient"] == patient_name, "Var_Slice_Skew"] = (
                np.var(slice_skews)
            )

            del scan, metadata, volume, slice_skews
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
/tmp/ipykernel_11/1887390073.py in <cell line: 0>()
     10 )
     11 
---> 12 df_train, df_test = image_data_preprocessor.create_image_features()
     13 
     14 print(

/tmp/ipykernel_11/239461681.py in create_image_features(self)
    151         print(f'Creating Image Features for Training Set\n{"-" * 40}')
    152         for i, patient_name in enumerate(self.train["Patient"].unique()):
--> 153             scan, metadata = self.load_scan("train", patient_name)
    154             scan_size = scan.nbytes >> 20
    155             print(

/tmp/ipykernel_11/239461681.py in load_scan(self, dataset, patient_name)
    125         )
    126         for i, s in enumerate(patient_directory):
--> 127             s_processed = self.crop(s.pixel_array)
    128             s_processed = self.resize(s_processed)
    129             s_processed = self.window(

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

    def train(self, X_train, y_train, df_train_ref=None):
        if df_train_ref is None:
            df_train_ref = X_train

        self.mlp_models = {f"CV{c}": [] for c in self.cv}
        self.qr_models = {f"CV{c}": [] for c in self.cv}

        models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
        for m in models:
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + (len(m)))}')
            for cv in self.cv:
                fold_col = f"CV{cv}_Fold"
                if fold_col not in X_train.columns:
                    raise KeyError(
                        f"Missing fold column {fold_col} in X_train. Ensure you did not drop it."
                    )
                for fold in sorted(X_train[fold_col].unique()):
                    trn_idx = X_train.loc[X_train[fold_col] != fold].index
                    val_idx = X_train.loc[X_train[fold_col] == fold].index

                    X_trn = X_train.loc[trn_idx, self.predictors].values
                    y_trn = y_train.loc[trn_idx].values.astype(float)
                    X_val = X_train.loc[val_idx, self.predictors].values
                    y_val = y_train.loc[val_idx].values.astype(float)

                    if m == "MLP":
                        mlp = MLPRegressor(
                            hidden_layer_sizes=(2**7, 2**7),
                            activation="relu",
                            alpha=0.0,
                            batch_size=self.mlp_parameters["batch_size"],
                            learning_rate_init=self.mlp_parameters["lr"],
                            max_iter=self.mlp_parameters["epochs"],
                            random_state=SEED,
                            shuffle=True,
                            early_stopping=False,
                            tol=0.0,
                            n_iter_no_change=self.mlp_parameters["epochs"] + 1,
                            verbose=False,
                        )
                        mlp.fit(X_trn, y_trn)
                        self.mlp_models[f"CV{cv}"].append(mlp)

                        fvc_pred = mlp.predict(X_val)
                        conf_pred = np.full_like(
                            fvc_pred,
                            np.maximum(np.std(y_trn - mlp.predict(X_trn)), 70.0),
                            dtype=float,
                        )

                        df_train_ref.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = (
                            fvc_pred
                        )
                        df_train_ref.loc[
                            val_idx, f"CV{cv}_MLP_Confidence_Predictions"
                        ] = conf_pred

                        oof_score_all = self.laplace_log_likelihood_metric(
                            y_val, fvc_pred, conf_pred
                        )
                        print(
                            f"CV {cv} {m} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - All Measurement Score: {oof_score_all:.6f}"
                        )

                    elif m == "QR":
                        qs = self.qr_parameters["quantiles"]
                        qr_models_fold = []
                        preds = []
                        for q in qs:
                            qr = QuantileRegressor(
                                quantile=q, alpha=0.0, solver="highs"
                            )
                            qr.fit(X_trn, y_trn)
                            qr_models_fold.append(qr)
                            preds.append(qr.predict(X_val))
                        self.qr_models[f"CV{cv}"].append(qr_models_fold)
                        preds = np.vstack(preds).T  # (n, 3)

                        for i, quantile in enumerate(qs):
                            df_train_ref.loc[
                                val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                            ] = preds[:, i]

                        fvc_pred = preds[:, 1]
                        conf_pred = np.maximum(preds[:, 2] - preds[:, 0], 70.0)
                        oof_score_all = self.laplace_log_likelihood_metric(
                            y_val, fvc_pred, conf_pred
                        )
                        print(
                            f"CV {cv} {m} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - All Measurement Score: {oof_score_all:.6f}"
                        )

    def predict(self, X_test):
        for cv in self.cv:
            if len(self.mlp_models[f"CV{cv}"]) > 0:
                mlp_preds = np.zeros((len(X_test),), dtype=float)
                for model in self.mlp_models[f"CV{cv}"]:
                    mlp_preds += model.predict(X_test[self.predictors].values) / len(
                        self.mlp_models[f"CV{cv}"]
                    )
                X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_preds
                X_test[f"CV{cv}_MLP_Confidence_Predictions"] = 200.0

            if len(self.qr_models[f"CV{cv}"]) > 0:
                qs = self.qr_parameters["quantiles"]
                qr_preds = np.zeros((len(X_test), len(qs)), dtype=float)
                for fold_models in self.qr_models[f"CV{cv}"]:
                    for i, qr in enumerate(fold_models):
                        qr_preds[:, i] += qr.predict(
                            X_test[self.predictors].values
                        ) / len(self.qr_models[f"CV{cv}"])
                for i, quantile in enumerate(qs):
                    X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_preds[:, i]




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
    "mlp_parameters": {"lr": 0.0005, "epochs": 200, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.0005,
        "epochs": 800,
        "batch_size": 2**5,
    },
}

missing = [c for c in model_parameters["predictors"] if c not in df_train.columns]
if missing:
    raise KeyError(f"Missing predictors in df_train: {missing}")

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train, df_train_ref=df_train)
qr_mlp.predict(df_test)

print(
    "Train prediction columns created:",
    [c for c in df_train.columns if "Predictions" in c][:10],
)
print(
    "Test prediction columns created:",
    [c for c in df_test.columns if "Predictions" in c][:10],
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1202047218.py in <cell line: 0>()
     41 missing = [c for c in model_parameters["predictors"] if c not in df_train.columns]
     42 if missing:
---> 43     raise KeyError(f"Missing predictors in df_train: {missing}")
     44 
     45 qr_mlp = QuantileRegressorMLP(**model_parameters)

KeyError: "Missing predictors in df_train: ['Scan_Skew', 'Scan_Kurtosis', 'Scan_Mean', 'Scan_Std', 'Scan_Var', 'Scan_Min_Volume', 'Scan_Max_Volume', 'Std_Slice_Skew', 'Var_Slice_Skew']"

## === cell 8
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
        return float(np.mean(score))

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
                    f"Missing required columns for {model}: {fvc_col}, {conf_col}"
                )

            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"], self.df_train[fvc_col], self.df_train[conf_col]
            )
            print(f"Single Model {model} Score: {score:.6f}")
            self.df_test["FVC"] = self.df_test[fvc_col]
            self.df_test["Confidence"] = self.df_test[conf_col]

        elif model.split("_")[1] == "QR":
            qs = [0.25, 0.5, 0.75]
            q0, q1, q2 = [f"{model}_{q}_Predictions" for q in qs]
            for c in [q0, q1, q2]:
                if c not in self.df_train.columns:
                    raise KeyError(f"Missing required column for {model}: {c}")

            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[q1],
                (self.df_train[q2] - self.df_train[q0]),
            )
            print(f"Single Model {model} Score: {score:.6f}")
            self.df_test["FVC"] = self.df_test[q1]
            self.df_test["Confidence"] = self.df_test[q2] - self.df_test[q0]
        else:
            raise ValueError("model must be like 'CV1_MLP' or 'CV1_QR'")

        self.df_test["Confidence"] = self.df_test["Confidence"].astype(float).abs()
        self.df_test["Confidence"] = np.maximum(self.df_test["Confidence"].values, 70.0)
        self.df_test["FVC"] = self.df_test["FVC"].astype(float)

        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 9
sub = SubmissionPipeline(df_train, df_test)

df_out = sub.single_model(model="CV1_QR")

df_out.to_csv("submission.csv", index=False)
print(df_out.head())
print("\nWrote submission.csv with shape:", df_out.shape)
print("Columns:", df_out.columns.tolist())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2997949622.py in <cell line: 0>()
      2 
      3 # Use CV1_QR by default (matches the original intent in the notebook cell 11)
----> 4 df_out = sub.single_model(model="CV1_QR")
      5 
      6 # Ensure exact submission format and file suffix

/tmp/ipykernel_11/4085788977.py in single_model(self, model)
     44             for c in [q0, q1, q2]:
     45                 if c not in self.df_train.columns:
---> 46                     raise KeyError(f"Missing required column for {model}: {c}")
     47 
     48             score = self.laplace_log_likelihood_metric(

KeyError: 'Missing required column for CV1_QR: CV1_QR_0.25_Predictions'

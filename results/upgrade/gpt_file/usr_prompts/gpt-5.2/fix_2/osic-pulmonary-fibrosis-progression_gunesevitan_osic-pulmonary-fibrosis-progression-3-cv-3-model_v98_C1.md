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

-6.924256046366523

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
from sklearn.linear_model import LinearRegression

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


BASE_PATH_CANDIDATES = [
    "../input/osic-pulmonary-fibrosis-progression",
    "/kaggle/input/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/osic-pulmonary-fibrosis-progression/osic-pulmonary-fibrosis-progression",
]
BASE_PATH = None
for p in BASE_PATH_CANDIDATES:
    if os.path.exists(p):
        BASE_PATH = p
        break
if BASE_PATH is None:
    raise FileNotFoundError("Could not find OSIC dataset folder in expected locations.")



## === cell 1
df_train = pd.read_csv(f"{BASE_PATH}/train.csv")
df_test = pd.read_csv(f"{BASE_PATH}/test.csv")
df_submission = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

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
            fvc_last2 = self.train[(self.train["Patient"] == patient_name)][
                "FVC"
            ].values[-2:]
            weeks_last2 = (
                self.train[(self.train["Patient"] == patient_name)]["Weeks"]
                .values[-2:]
                .reshape(-1, 1)
            )
            std = fvc_last2.std()
            if std == 0:
                z = np.zeros_like(fvc_last2, dtype=np.float32)
            else:
                z = (fvc_last2 - fvc_last2.mean()) / std

            reg = LinearRegression().fit(weeks_last2, z)

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
        max_slices_per_patient=24,
    ):
        self.train = train.copy(deep=True)
        self.test = test.copy(deep=True)

        self.resize_shape = resize_shape

        self.window_width = window_width
        self.window_center = window_center
        self.y_min = y_min
        self.y_max = y_max

        self.scale = scale
        self.max_slices_per_patient = max_slices_per_patient

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

    def _list_dcm_files(self, dataset, patient_name):
        folder = f"{BASE_PATH}/{dataset}/{patient_name}"
        files = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]
        if not files:
            return []

        def key_fn(fn):
            stem = os.path.splitext(fn)[0]
            try:
                return int(stem)
            except Exception:
                return stem

        files = sorted(files, key=key_fn)
        return [os.path.join(folder, f) for f in files]

    def _select_subset(self, file_paths):
        n = len(file_paths)
        if n <= self.max_slices_per_patient:
            return file_paths
        idx = np.linspace(0, n - 1, self.max_slices_per_patient).round().astype(int)
        idx = np.unique(idx)
        return [file_paths[i] for i in idx]

    def load_scan_sampled(self, dataset, patient_name):
        file_paths = self._list_dcm_files(dataset, patient_name)
        file_paths = self._select_subset(file_paths)
        if len(file_paths) == 0:
            return np.zeros(
                (0, self.resize_shape[0], self.resize_shape[1]), dtype=np.int16
            ), {"PixelSpacing": [np.nan, np.nan], "SliceSpacing": np.nan}

        slices = []
        pixel_spacings = []
        slice_positions = []

        for fp in file_paths:
            ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
            try:
                posz = float(ds.ImagePositionPatient[2])
            except Exception:
                try:
                    posz = float(ds.InstanceNumber)
                except Exception:
                    posz = np.nan
            slice_positions.append(posz)
            try:
                pixel_spacings.append(np.array(ds.PixelSpacing, dtype=np.float32))
            except Exception:
                pixel_spacings.append(np.array([np.nan, np.nan], dtype=np.float32))

            arr = ds.pixel_array.astype(np.int16)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = self.crop(arr)
            arr = self.resize(arr)
            arr = self.window(
                arr,
                slope,
                intercept,
                self.window_width,
                self.window_center,
                self.y_min,
                self.y_max,
            )
            if np.all(arr == 0):
                continue
            slices.append(arr.astype(np.int16))

        if len(slices) == 0:
            scan = np.zeros(
                (0, self.resize_shape[0], self.resize_shape[1]), dtype=np.int16
            )
        else:
            scan = np.stack(slices, axis=0)

        pixel_spacings = np.stack(pixel_spacings, axis=0)
        metadata = {}
        metadata["PixelSpacing"] = list(np.round(np.nanmean(pixel_spacings, axis=0), 3))

        slice_positions = np.array(slice_positions, dtype=np.float32)
        slice_positions = slice_positions[~np.isnan(slice_positions)]
        if patient_name == "ID00128637202219474716089":
            metadata["SliceSpacing"] = 5.0
        elif patient_name == "ID00132637202222178761324":
            metadata["SliceSpacing"] = 0.7
        else:
            if len(slice_positions) >= 2:
                diffs = np.abs(np.diff(np.sort(np.round(slice_positions, 3))))
                diffs = diffs[diffs > 0]
                if len(diffs) == 0:
                    metadata["SliceSpacing"] = 1.0
                else:
                    metadata["SliceSpacing"] = float(mode(diffs, keepdims=True).mode[0])
            else:
                metadata["SliceSpacing"] = 1.0

        return scan, metadata

    def _add_scan_features(self, df, dataset_name):
        for i, patient_name in enumerate(df["Patient"].unique()):
            scan, metadata = self.load_scan_sampled(dataset_name, patient_name)
            if scan.shape[0] == 0:
                for col in [
                    "VoxelVolume",
                    "Scan_Skew",
                    "Scan_Kurtosis",
                    "Scan_Mean",
                    "Scan_Std",
                    "Scan_Var",
                    "Scan_Min_Volume",
                    "Scan_Max_Volume",
                    "Std_Slice_Skew",
                    "Var_Slice_Skew",
                ]:
                    df.loc[df["Patient"] == patient_name, col] = 0.0
                continue

            volume = (
                (metadata["SliceSpacing"] * scan.shape[0])
                * (metadata["PixelSpacing"][0] * scan.shape[1])
                * (metadata["PixelSpacing"][1] * scan.shape[2])
            )
            voxel_vol = volume / (scan.shape[0] * scan.shape[1] * scan.shape[2])
            flat = scan.reshape(-1).astype(np.float32)

            df.loc[df["Patient"] == patient_name, "VoxelVolume"] = voxel_vol
            df.loc[df["Patient"] == patient_name, "Scan_Skew"] = float(skew(flat))
            df.loc[df["Patient"] == patient_name, "Scan_Kurtosis"] = float(
                kurtosis(flat)
            )
            df.loc[df["Patient"] == patient_name, "Scan_Mean"] = float(flat.mean())
            df.loc[df["Patient"] == patient_name, "Scan_Std"] = float(flat.std())
            df.loc[df["Patient"] == patient_name, "Scan_Var"] = float(flat.var())
            df.loc[df["Patient"] == patient_name, "Scan_Min_Volume"] = float(
                (flat == self.y_min).sum() * voxel_vol
            )
            df.loc[df["Patient"] == patient_name, "Scan_Max_Volume"] = float(
                (flat == self.y_max).sum() * voxel_vol
            )

            slice_skews = [skew(s.reshape(-1).astype(np.float32)) for s in scan]
            df.loc[df["Patient"] == patient_name, "Std_Slice_Skew"] = float(
                np.std(slice_skews)
            )
            df.loc[df["Patient"] == patient_name, "Var_Slice_Skew"] = float(
                np.var(slice_skews)
            )

            if (i + 1) % 10 == 0 or (i + 1) == df["Patient"].nunique():
                print(
                    f'[{i + 1}/{df["Patient"].nunique()}] Processed {dataset_name} patient {patient_name} scan_shape={scan.shape}'
                )

            del scan, metadata, flat, slice_skews
            gc.collect()

    def create_image_features(self):
        print(f'Creating Image Features for Training Set\n{"-" * 40}')
        self._add_scan_features(self.train, "train")

        print(f'\nCreating Image Features for Test Set\n{"-" * 36}')
        self._add_scan_features(self.test, "test")

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
            scaler.fit(self.train.loc[:, scale_features].astype(np.float32))
            self.train.loc[:, scale_features] = scaler.transform(
                self.train.loc[:, scale_features].astype(np.float32)
            )
            self.test.loc[:, scale_features] = scaler.transform(
                self.test.loc[:, scale_features].astype(np.float32)
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
    max_slices_per_patient=24,
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
/tmp/ipykernel_11/4056710223.py in <cell line: 0>()
     11 )
     12 
---> 13 df_train, df_test = image_data_preprocessor.create_image_features()
     14 
     15 print(

/tmp/ipykernel_11/1665389558.py in create_image_features(self)
    221     def create_image_features(self):
    222         print(f'Creating Image Features for Training Set\n{"-" * 40}')
--> 223         self._add_scan_features(self.train, "train")
    224 
    225         print(f'\nCreating Image Features for Test Set\n{"-" * 36}')

/tmp/ipykernel_11/1665389558.py in _add_scan_features(self, df, dataset_name)
    162     def _add_scan_features(self, df, dataset_name):
    163         for i, patient_name in enumerate(df["Patient"].unique()):
--> 164             scan, metadata = self.load_scan_sampled(dataset_name, patient_name)
    165             if scan.shape[0] == 0:
    166                 # Safe defaults if scan couldn't be read

/tmp/ipykernel_11/1665389558.py in load_scan_sampled(self, dataset, patient_name)
    112                 pixel_spacings.append(np.array([np.nan, np.nan], dtype=np.float32))
    113 
--> 114             arr = ds.pixel_array.astype(np.int16)
    115             slope = float(getattr(ds, "RescaleSlope", 1.0))
    116             intercept = float(getattr(ds, "RescaleIntercept", 0.0))

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
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, GaussianDropout


def seed_tf(seed):
    tf.random.set_seed(seed)


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

    def get_model(self, input_shape, m):
        model = None

        if m == "MLP":
            input_layer = Input(shape=(input_shape,))
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
            input_layer = Input(shape=(input_shape,))
            x = Dense(2**7, activation="relu")(input_layer)
            x = GaussianDropout(0.01)(x)
            x = Dense(2**7, activation="relu")(x)
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

    def train(self, X_train, y_train, df_train_ref):

        self.mlp_oof = pd.DataFrame(np.zeros((len(y_train), 2)))
        self.qr_oof = pd.DataFrame(
            np.zeros((len(y_train), len(self.qr_parameters["quantiles"])))
        )

        self.mlp_models = {"CV1": [], "CV2": [], "CV3": []}
        self.qr_models = {"CV1": [], "CV2": [], "CV3": []}

        models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
        for m in models:
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + (len(m)))}')

            for cv in range(1, 4):
                for fold in sorted(X_train[f"CV{cv}_Fold"].unique()):

                    trn_idx = X_train.loc[X_train[f"CV{cv}_Fold"] != fold].index
                    val_idx = X_train.loc[X_train[f"CV{cv}_Fold"] == fold].index

                    X_trn = X_train.loc[trn_idx, self.predictors]
                    y_trn = y_train.loc[trn_idx]
                    X_val = X_train.loc[val_idx, self.predictors]
                    y_val = y_train.loc[val_idx]

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

                        fold_final_scores = []
                        grp = (
                            df_train_ref.loc[val_idx]
                            .groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index()
                        )
                        for df_patient in np.array_split(
                            grp, df_train_ref.loc[val_idx, "Patient"].nunique()
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
                            df_train_ref.loc[
                                val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                            ] = predictions[:, i]

                        fold_final_scores = []
                        grp = (
                            df_train_ref.loc[val_idx]
                            .groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index()
                        )
                        q = self.qr_parameters["quantiles"]
                        for df_patient in np.array_split(
                            grp, df_train_ref.loc[val_idx, "Patient"].nunique()
                        ):
                            fold_final_scores.append(
                                self.laplace_log_likelihood_metric(
                                    df_patient["FVC"],
                                    df_patient[f"CV{cv}_QR_{q[1]}_Predictions"],
                                    (
                                        df_patient[f"CV{cv}_QR_{q[2]}_Predictions"]
                                        - df_patient[f"CV{cv}_QR_{q[0]}_Predictions"]
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
                    grp = (
                        df_train_ref.groupby("Patient").nth([-1, -2, -3]).reset_index()
                    )
                    for df_patient in np.array_split(
                        grp, df_train_ref["Patient"].nunique()
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
                elif m == "QR":
                    q = self.qr_parameters["quantiles"]
                    grp = (
                        df_train_ref.groupby("Patient").nth([-1, -2, -3]).reset_index()
                    )
                    for df_patient in np.array_split(
                        grp, df_train_ref["Patient"].nunique()
                    ):
                        oof_final_scores.append(
                            self.laplace_log_likelihood_metric(
                                df_patient["FVC"],
                                df_patient[f"CV{cv}_QR_{q[1]}_Predictions"],
                                (
                                    df_patient[f"CV{cv}_QR_{q[2]}_Predictions"]
                                    - df_patient[f"CV{cv}_QR_{q[0]}_Predictions"]
                                ),
                            )
                        )
                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_train,
                        self.qr_oof.iloc[:, 1],
                        (self.qr_oof.iloc[:, 2] - self.qr_oof.iloc[:, 0]),
                    )

                print(
                    f'{"-" * 30}\nCV {cv} {m} All Measurement OOF Score {oof_all_score:.6} - Final 3 Measurement OOF Score {np.mean(oof_final_scores):.6} [Std: {np.std(oof_final_scores):.6}]\n{"-" * 30}\n'
                )

    def predict(self, X_test):
        for cv in range(1, 4):
            mlp_predictions = np.zeros((len(X_test), 2), dtype=np.float32)
            for model in self.mlp_models[f"CV{cv}"]:
                mlp_predictions += model.predict(
                    X_test[self.predictors], verbose=0
                ) / len(self.mlp_models[f"CV{cv}"])

            X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_predictions[:, 0]
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_predictions[:, 1]

            qr_predictions = np.zeros(
                (len(X_test), len(self.qr_parameters["quantiles"])), dtype=np.float32
            )
            for model in self.qr_models[f"CV{cv}"]:
                qr_predictions += model.predict(
                    X_test[self.predictors], verbose=0
                ) / len(self.qr_models[f"CV{cv}"])

            for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_predictions[:, i]




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
seed_everything(SEED)
seed_tf(SEED)

X_train = df_train.drop(columns=["FVC", "Weeks"])
y_train = df_train[["FVC"]].copy(deep=True)

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
        "quantiles": [0.15, 0.5, 0.85],
        "lr": 0.0005,
        "epochs": 150,
        "batch_size": 2**5,
    },
}

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train, df_train_ref=df_train)
qr_mlp.predict(df_test)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3565209051.py in <cell line: 0>()
     26 
     27 qr_mlp = QuantileRegressorMLP(**model_parameters)
---> 28 qr_mlp.train(X_train, y_train, df_train_ref=df_train)
     29 qr_mlp.predict(df_test)
     30 

/tmp/ipykernel_11/2970889468.py in train(self, X_train, y_train, df_train_ref)
    209                             )
    210 
--> 211                     oof_score_all = self.laplace_log_likelihood_metric(
    212                         y_val, oof_predictions, oof_confidence
    213                     )

/tmp/ipykernel_11/2970889468.py in laplace_log_likelihood_metric(self, y_true, y_pred, sigma)
     24 
     25         sigma_clipped = np.maximum(sigma, 70)
---> 26         delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
     27         score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
     28             np.sqrt(2) * sigma_clipped

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __sub__(self, other)
    192     @unpack_zerodim_and_defer("__sub__")
    193     def __sub__(self, other):
--> 194         return self._arith_method(other, operator.sub)
    195 
    196     @unpack_zerodim_and_defer("__rsub__")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _arith_method(self, other, op)
   7908         other = ops.maybe_prepare_scalar_for_op(other, (self.shape[axis],))
   7909 
-> 7910         self, other = self._align_for_op(other, axis, flex=True, level=None)
   7911 
   7912         with np.errstate(all="ignore"):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _align_for_op(self, other, axis, flex, level)
   8139         if isinstance(right, np.ndarray):
   8140             if right.ndim == 1:
-> 8141                 right = to_series(right)
   8142 
   8143             elif right.ndim == 2:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_series(right)
   8131             else:
   8132                 if len(left.columns) != len(right):
-> 8133                     raise ValueError(
   8134                         msg.format(req_len=len(left.columns), given_len=len(right))
   8135                     )

ValueError: Unable to coerce to Series, length must be 1: given 692

## === cell 8
class SubmissionPipeline:

    def __init__(self, df_train, df_test, quantiles=(0.15, 0.5, 0.85)):
        self.df_train = df_train
        self.df_test = df_test
        self.quantiles = list(quantiles)

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
            q = self.quantiles
            for df in [self.df_train, self.df_test]:
                df[f"CV{cv}_FVC"] = (df[f"CV{cv}_MLP_FVC_Predictions"] * 0.5) + (
                    df[f"CV{cv}_QR_{q[1]}_Predictions"] * 0.5
                )
                df[f"CV{cv}_Confidence"] = (
                    df[f"CV{cv}_MLP_Confidence_Predictions"] * 0.5
                ) + (
                    (
                        df[f"CV{cv}_QR_{q[2]}_Predictions"]
                        - df[f"CV{cv}_QR_{q[0]}_Predictions"]
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
                        self.df_train[f"CV{cv}_QR_{q[1]}_Predictions"],
                        (
                            self.df_train[f"CV{cv}_QR_{q[2]}_Predictions"]
                            - self.df_train[f"CV{cv}_QR_{q[0]}_Predictions"]
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

        self.df_test["Confidence"] = np.maximum(
            self.df_test["Confidence"].values.astype(np.float32), 70.0
        )
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 9
sub = SubmissionPipeline(
    df_train, df_test, quantiles=tuple(model_parameters["qr_parameters"]["quantiles"])
)
df_submission_out = sub.blend(by="cv", model=None, cv=1)

sample = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")
df_submission_out = sample[["Patient_Week"]].merge(
    df_submission_out, on="Patient_Week", how="left"
)

if df_submission_out[["FVC", "Confidence"]].isna().any().any():
    baseline_map = (
        df_test.drop_duplicates("Patient")
        .set_index("Patient")["FVC_Baseline"]
        .to_dict()
    )
    tmp_patient = df_submission_out["Patient_Week"].str.split("_", n=1, expand=True)[0]
    df_submission_out["FVC"] = (
        df_submission_out["FVC"]
        .fillna(tmp_patient.map(baseline_map))
        .astype(np.float32)
    )
    df_submission_out["Confidence"] = (
        df_submission_out["Confidence"].fillna(200.0).astype(np.float32)
    )

df_submission_out.to_csv("submission.csv", index=False)
print(df_submission_out.head())
print("Wrote submission.csv with shape:", df_submission_out.shape)



## --- ERROR in cell 9, traceback:
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

KeyError: 'CV1_QR_0.5_Predictions'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2010659047.py in <cell line: 0>()
      2     df_train, df_test, quantiles=tuple(model_parameters["qr_parameters"]["quantiles"])
      3 )
----> 4 df_submission_out = sub.blend(by="cv", model=None, cv=1)
      5 
      6 # Align with sample_submission ordering to avoid any potential row-order issues.

/tmp/ipykernel_11/439176973.py in blend(self, by, model, cv)
     26             for df in [self.df_train, self.df_test]:
     27                 df[f"CV{cv}_FVC"] = (df[f"CV{cv}_MLP_FVC_Predictions"] * 0.5) + (
---> 28                     df[f"CV{cv}_QR_{q[1]}_Predictions"] * 0.5
     29                 )
     30                 df[f"CV{cv}_Confidence"] = (

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

KeyError: 'CV1_QR_0.5_Predictions'

## === cell 10
df_submission_out

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3922690370.py in <cell line: 0>()
----> 1 df_submission_out

NameError: name 'df_submission_out' is not defined

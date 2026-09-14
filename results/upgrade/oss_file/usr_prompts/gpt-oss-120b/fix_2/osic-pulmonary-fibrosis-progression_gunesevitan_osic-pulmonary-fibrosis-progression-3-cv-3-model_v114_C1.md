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

try:
    import pydicom
except Exception:  # pragma: no cover
    pydicom = None

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
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import Callback

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)




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
            patient_data = self.train[self.train["Patient"] == patient_name]
            if len(patient_data) < 2:
                z = np.zeros(2)
            else:
                last_two = patient_data["FVC"].values[-2:]
                z = (last_two - last_two.mean()) / (
                    last_two.std() if last_two.std() != 0 else 1
                )
            reg = LinearRegression().fit(
                patient_data["Weeks"].values[-2:].reshape(-1, 1), z
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
            scale_features = ["FVC_Baseline"]
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
        if pydicom is None:
            raise ImportError("pydicom is not available in this environment.")
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
                continue

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
            scan[i] = np.int16(s_processed)

        del patient_directory
        scan = scan[~np.all(scan == 0, axis=(-1, -2))]
        return scan, metadata

    def create_image_features(self):
        print(f'Creating Image Features for Training Set\n{"-" * 40}')
        try:
            df_train_features = pd.read_csv(
                "../input/osic-pulmonary-fibrosis-progression-features/df_scan_features.csv"
            )
            self.train = self.train.merge(df_train_features, on="Patient", how="left")
        except Exception:
            pass
        print(f'\nCreating Image Features for Test Set\n{"-" * 36}')
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
    scale=True,
)

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
        sigma = y_pred[:, 1]
        fvc_pred = y_pred[:, 0]
        sigma_clipped = K.maximum(sigma, K.constant(70, dtype="float32"))
        delta = K.abs(y_true[:, 0] - fvc_pred)
        delta_clipped = K.minimum(delta, K.constant(1000, dtype="float32"))
        score = (delta_clipped / sigma_clipped) * K.sqrt(K.constant(2.0)) + K.log(
            sigma_clipped * K.sqrt(K.constant(2.0))
        )
        return K.mean(score)

    def tilted_loss(self, y_true, y_pred):
        quantiles = K.constant(
            np.array([self.qr_parameters["quantiles"]]), dtype="float32"
        )
        error = y_true - y_pred
        return K.mean(K.maximum(quantiles * error, (quantiles - 1) * error))

    def get_model(self, input_shape, m):
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
                optimizer=Adam(lr=self.mlp_parameters["lr"]),
                metrics=[self.laplace_log_likelihood_loss],
            )
        elif m == "QR":
            input_layer = Input(shape=input_shape)
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
                optimizer=Adam(lr=self.qr_parameters["lr"]),
                metrics=[self.laplace_log_likelihood_loss],
            )
        else:
            model = None
        return model

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
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + len(m))}')
            for cv in self.cv:
                fold_col = f"CV{cv}_Fold"
                if fold_col not in X_train.columns:
                    X_train[fold_col] = 1  # simple single fold

                for fold in sorted(X_train[fold_col].unique()):
                    trn_idx = X_train.loc[X_train[fold_col] != fold].index
                    val_idx = X_train.loc[X_train[fold_col] == fold].index

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
                    else:  # QR
                        model.fit(
                            X_trn,
                            y_trn,
                            epochs=self.qr_parameters["epochs"],
                            batch_size=self.qr_parameters["batch_size"],
                            verbose=0,
                        )
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

                    else:  # QR
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
                            low = df_patient[
                                f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                            ]
                            med = df_patient[
                                f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                            ]
                            high = df_patient[
                                f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                            ]
                            fold_final_scores.append(
                                self.laplace_log_likelihood_metric(
                                    df_patient["FVC"], med, high - low
                                )
                            )

                    oof_score_all = self.laplace_log_likelihood_metric(
                        y_val, oof_predictions, oof_confidence
                    )
                    print(
                        f"CV {cv} {m} Fold {int(fold)} - All Measurement Score: {oof_score_all:.6f} - Final 3 Measurement Score {np.mean(fold_final_scores):.6f} [Std: {np.std(fold_final_scores):.6f}]"
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
                else:
                    for df_patient in np.array_split(
                        df_train.groupby("Patient").nth([-1, -2, -3]).reset_index(),
                        df_train["Patient"].nunique(),
                    ):
                        low = df_patient[
                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][0]}_Predictions'
                        ]
                        med = df_patient[
                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][1]}_Predictions'
                        ]
                        high = df_patient[
                            f'CV{cv}_QR_{self.qr_parameters["quantiles"][2]}_Predictions'
                        ]
                        oof_final_scores.append(
                            self.laplace_log_likelihood_metric(
                                df_patient["FVC"], med, high - low
                            )
                        )
                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_train,
                        self.qr_oof.iloc[:, 1],
                        self.qr_oof.iloc[:, 2] - self.qr_oof.iloc[:, 0],
                    )

                print(
                    f'{"-" * 30}\nCV {cv} {m} All Measurement OOF Score {oof_all_score:.6f} - Final 3 Measurement OOF Score {np.mean(oof_final_scores):.6f} [Std: {np.std(oof_final_scores):.6f}]\n{"-" * 30}\n'
                )

    def predict(self, X_test):
        for cv in self.cv:
            mlp_predictions = np.zeros((len(X_test), 2))
            for model in self.mlp_models.get(f"CV{cv}", []):
                mlp_predictions += model.predict(X_test[self.predictors]) / len(
                    self.mlp_models[f"CV{cv}"]
                )
            if mlp_predictions.shape[1] == 2:
                X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_predictions[:, 0]
                X_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_predictions[:, 1]

            qr_predictions = np.zeros(
                (len(X_test), len(self.qr_parameters["quantiles"]))
            )
            for model in self.qr_models.get(f"CV{cv}", []):
                qr_predictions += model.predict(X_test[self.predictors]) / len(
                    self.qr_models[f"CV{cv}"]
                )
            for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_predictions[:, i]


qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train)
qr_mlp.predict(df_test)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2439034644.py in <cell line: 0>()
    287 
    288 qr_mlp = QuantileRegressorMLP(**model_parameters)
--> 289 qr_mlp.train(X_train, y_train)
    290 qr_mlp.predict(df_test)
    291 

/tmp/ipykernel_11/2439034644.py in train(self, X_train, y_train)
    130                     y_val = y_train.loc[val_idx]
    131 
--> 132                     model = self.get_model(input_shape=X_trn.shape[1], m=m)
    133                     if m == "MLP":
    134                         model.fit(

/tmp/ipykernel_11/2439034644.py in get_model(self, input_shape, m)
     65     def get_model(self, input_shape, m):
     66         if m == "MLP":
---> 67             input_layer = Input(shape=input_shape)
     68             x = Dense(2**7, activation="relu")(input_layer)
     69             x = GaussianDropout(0.01)(x)

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/input_layer.py in Input(shape, batch_size, dtype, sparse, batch_shape, name, tensor, optional)
    189     ```
    190     """
--> 191     layer = InputLayer(
    192         shape=shape,
    193         batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/input_layer.py in __init__(self, shape, batch_size, dtype, sparse, batch_shape, input_tensor, optional, name, **kwargs)
     90 
     91             if shape is not None:
---> 92                 shape = backend.standardize_shape(shape)
     93                 batch_shape = (batch_size,) + shape
     94 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in standardize_shape(shape)
    560             raise ValueError("Undefined shapes are not supported.")
    561         if not hasattr(shape, "__iter__"):
--> 562             raise ValueError(f"Cannot convert '{shape}' to a shape.")
    563         if config.backend() == "tensorflow":
    564             if isinstance(shape, tf.TensorShape):

ValueError: Cannot convert '9' to a shape.

## === cell 7
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
        if len(prediction_cols) < 2:
            raise ValueError(f"Not enough prediction columns for model {model}")
        if model.split("_")[1] == "MLP":
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[prediction_cols[0]],
                self.df_train[prediction_cols[1]],
            )
            print(f"Single Model {model} Score: {score:.6f}")
            self.df_test["FVC"] = self.df_test[prediction_cols[0]]
            self.df_test["Confidence"] = self.df_test[prediction_cols[1]]
        else:  # QR
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[prediction_cols[1]],
                (self.df_train[prediction_cols[2]] - self.df_train[prediction_cols[0]]),
            )
            print(f"Single Model {model} Score: {score:.6f}")
            self.df_test["FVC"] = self.df_test[prediction_cols[1]]
            self.df_test["Confidence"] = (
                self.df_test[prediction_cols[2]] - self.df_test[prediction_cols[0]]
            )
        print(f'\n{self.df_test[["Patient_Week", "FVC", "Confidence"]].describe()}')
        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 8
sub = SubmissionPipeline(df_train, df_test)
df_submission = sub.single_model(model="CV1_MLP")
df_submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3651998566.py in <cell line: 0>()
      1 sub = SubmissionPipeline(df_train, df_test)
----> 2 df_submission = sub.single_model(model="CV1_MLP")
      3 df_submission.to_csv("submission.csv", index=False)
      4 

/tmp/ipykernel_11/4051242675.py in single_model(self, model)
     23         ]
     24         if len(prediction_cols) < 2:
---> 25             raise ValueError(f"Not enough prediction columns for model {model}")
     26         if model.split("_")[1] == "MLP":
     27             score = self.laplace_log_likelihood_metric(

ValueError: Not enough prediction columns for model CV1_MLP

## === cell 9
df_submission

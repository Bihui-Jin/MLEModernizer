# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_C_DESCRIPTORS", "0")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import pickle
import random
import gc
import warnings

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from scipy.stats import skew, mode, kurtosis

import matplotlib.pyplot as plt

try:
    import cv2
except Exception:
    cv2 = None

pydicom = None

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
from tensorflow.keras.optimizers import Adam

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)
warnings.filterwarnings("ignore")

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass




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
            std = np.std(last2)
            if std == 0:
                z = last2 - np.mean(last2)
            else:
                z = (last2 - np.mean(last2)) / std

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
        pw = self.submission["Patient_Week"].str.split("_", n=1, expand=True)
        self.submission["Patient"] = pw[0].astype(str)
        self.submission["Weeks"] = pw[1].astype(np.int16)
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

        df_train_out = df_all.loc[df_all["Type"] == "Train", :].drop(columns=["Type"])
        for i in range(1, 4):
            df_train_out[f"CV{i}_Fold"] = df_train_out[f"CV{i}_Fold"].astype(np.uint8)
        df_test_out = df_all.loc[df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

        return df_train_out.copy(deep=True), df_test_out.reset_index(drop=True).copy(
            deep=True
        )




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
        if cv2 is None:
            return s
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

    def _list_unique_sorted_dicoms(self, base_path):
        global pydicom
        if pydicom is None:
            import pydicom as _pydicom

            pydicom = _pydicom

        files = os.listdir(base_path)
        headers = []
        for fn in files:
            try:
                ds = pydicom.dcmread(f"{base_path}/{fn}", stop_before_pixels=True)
                headers.append((fn, ds))
            except Exception:
                continue
        if len(headers) == 0:
            return []

        try:
            headers.sort(key=lambda t: float(t[1].ImagePositionPatient[2]))
            keys = [float(t[1].ImagePositionPatient[2]) for t in headers]
        except Exception:
            headers.sort(key=lambda t: int(t[1].InstanceNumber))
            keys = [int(t[1].InstanceNumber) for t in headers]

        seen = set()
        out = []
        for (fn, ds), k in zip(headers, keys):
            k_round = round(float(k), 4) if isinstance(k, float) else int(k)
            if k_round in seen:
                continue
            seen.add(k_round)
            out.append((fn, ds))
        return out

    def load_scan(self, dataset, patient_name):
        global pydicom
        if pydicom is None:
            import pydicom as _pydicom

            pydicom = _pydicom

        base_path = (
            f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}"
        )

        kept = self._list_unique_sorted_dicoms(base_path)
        if len(kept) == 0:
            return np.zeros(
                (0, self.resize_shape[0], self.resize_shape[1]), dtype=np.int16
            ), {
                "PixelSpacing": [np.nan, np.nan],
                "SliceSpacing": np.nan,
            }

        pixel_spacings = np.zeros((len(kept), 2), dtype=np.float32)
        slice_positions = np.zeros((len(kept)), dtype=np.float32)

        for i, (_, hdr) in enumerate(kept):
            try:
                pixel_spacings[i, :] = np.array(hdr.PixelSpacing, dtype=np.float32)
            except Exception:
                pixel_spacings[i, :] = np.nan
            try:
                slice_positions[i] = float(hdr.ImagePositionPatient[2])
            except Exception:
                slice_positions[i] = np.nan

        metadata = {}
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
                metadata["SliceSpacing"] = np.nan
            else:
                metadata["SliceSpacing"] = float(mode(diffs, keepdims=False)[0])

        scan = np.zeros(
            (len(kept), self.resize_shape[0], self.resize_shape[1]),
            dtype=np.int16,
        )

        for i, (fn, hdr) in enumerate(kept):
            try:
                ds = pydicom.dcmread(f"{base_path}/{fn}")
            except Exception:
                continue

            s_processed = self.crop(ds.pixel_array)
            s_processed = self.resize(s_processed)
            s_processed = self.window(
                s_processed,
                ds.RescaleSlope,
                ds.RescaleIntercept,
                self.window_width,
                self.window_center,
                self.y_min,
                self.y_max,
            )

            if np.all(s_processed == 0):
                continue
            scan[i] = np.int16(s_processed)

        scan = scan[~np.all(scan == 0, axis=(-1, -2))]
        return scan, metadata

    def load_scan_metadata_only(self, dataset, patient_name):
        global pydicom
        if pydicom is None:
            import pydicom as _pydicom

            pydicom = _pydicom

        base_path = (
            f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}"
        )
        kept = self._list_unique_sorted_dicoms(base_path)
        if len(kept) == 0:
            raise RuntimeError("No readable DICOM headers found")

        try:
            slice_positions = np.array(
                [float(ds.ImagePositionPatient[2]) for _, ds in kept], dtype=np.float32
            )
        except Exception:
            slice_positions = np.arange(len(kept), dtype=np.float32)

        pixel_spacings = []
        for _, s in kept:
            try:
                pixel_spacings.append(
                    [float(s.PixelSpacing[0]), float(s.PixelSpacing[1])]
                )
            except Exception:
                continue
        if len(pixel_spacings) == 0:
            px = [np.nan, np.nan]
        else:
            px = np.nanmean(np.array(pixel_spacings, dtype=np.float32), axis=0).tolist()

        if patient_name == "ID00128637202219474716089":
            slice_spacing = 5.0
        elif patient_name == "ID00132637202222178761324":
            slice_spacing = 0.7
        else:
            diffs = np.abs(np.diff(np.round(slice_positions, 3)))
            if diffs.size == 0:
                slice_spacing = np.nan
            else:
                slice_spacing = float(mode(diffs, keepdims=False)[0])

        try:
            rows = int(kept[0][1].Rows)
            cols = int(kept[0][1].Columns)
        except Exception:
            rows, cols = self.resize_shape[0], self.resize_shape[1]

        metadata = {
            "PixelSpacing": [float(px[0]), float(px[1])],
            "SliceSpacing": (
                float(slice_spacing) if slice_spacing == slice_spacing else np.nan
            ),
            "NumSlices": int(len(kept)),
            "Rows": int(rows),
            "Cols": int(cols),
        }
        return metadata

    def _add_empty_image_feature_columns(self, df):
        image_cols = [
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
        for c in image_cols:
            if c not in df.columns:
                df[c] = 0.0
        return df

    def _scan_stats(self, scan):
        flat = scan.reshape(-1).astype(np.float32, copy=False)
        scan_skew = float(skew(flat))
        scan_kurt = float(kurtosis(flat))
        scan_mean = float(flat.mean())
        scan_std = float(flat.std())
        scan_var = float(flat.var())

        ss = scan.reshape(scan.shape[0], -1).astype(np.float32, copy=False)
        slice_skews = np.empty((ss.shape[0],), dtype=np.float32)
        for i in range(ss.shape[0]):
            slice_skews[i] = skew(ss[i])
        std_slice_skew = float(slice_skews.std())
        var_slice_skew = float(slice_skews.var())
        return (
            scan_skew,
            scan_kurt,
            scan_mean,
            scan_std,
            scan_var,
            std_slice_skew,
            var_slice_skew,
        )

    def create_image_features(self):
        print(f'Creating Image Features for Training Set\n{"-" * 40}')
        imported = False
        try:
            df_train_features = pd.read_csv(
                "../input/osic-pulmonary-fibrosis-progression-features/df_scan_features.csv"
            )
            self.train = self.train.merge(df_train_features, on="Patient", how="left")
            print("Image Features Imported")
            imported = True
        except Exception as e:
            print(
                f"Precomputed feature file not found, computing from DICOMs. Reason: {repr(e)}"
            )

        if not imported:
            try:
                uniq_train_patients = self.train["Patient"].unique()
                for i, patient_name in enumerate(uniq_train_patients):
                    scan, metadata = self.load_scan("train", patient_name)
                    scan_size = scan.nbytes >> 20
                    print(
                        f"[{i + 1}/{len(uniq_train_patients)}] {patient_name} Shape: {scan.shape} - Size: {scan_size} MB"
                    )

                    volume = (
                        (metadata["SliceSpacing"] * scan.shape[0])
                        * (metadata["PixelSpacing"][0] * scan.shape[1])
                        * (metadata["PixelSpacing"][1] * scan.shape[2])
                    )
                    voxel_vol = volume / (scan.shape[0] * scan.shape[1] * scan.shape[2])
                    self.train.loc[
                        self.train["Patient"] == patient_name, "VoxelVolume"
                    ] = voxel_vol

                    (
                        scan_skew,
                        scan_kurt,
                        scan_mean,
                        scan_std,
                        scan_var,
                        std_slice_skew,
                        var_slice_skew,
                    ) = self._scan_stats(scan)
                    self.train.loc[
                        self.train["Patient"] == patient_name, "Scan_Skew"
                    ] = scan_skew
                    self.train.loc[
                        self.train["Patient"] == patient_name, "Scan_Kurtosis"
                    ] = scan_kurt
                    self.train.loc[
                        self.train["Patient"] == patient_name, "Scan_Mean"
                    ] = scan_mean
                    self.train.loc[
                        self.train["Patient"] == patient_name, "Scan_Std"
                    ] = scan_std
                    self.train.loc[
                        self.train["Patient"] == patient_name, "Scan_Var"
                    ] = scan_var

                    self.train.loc[
                        self.train["Patient"] == patient_name, "Scan_Min_Volume"
                    ] = (float((scan == self.y_min).sum()) * voxel_vol)
                    self.train.loc[
                        self.train["Patient"] == patient_name, "Scan_Max_Volume"
                    ] = (float((scan == self.y_max).sum()) * voxel_vol)

                    self.train.loc[
                        self.train["Patient"] == patient_name, "Std_Slice_Skew"
                    ] = std_slice_skew
                    self.train.loc[
                        self.train["Patient"] == patient_name, "Var_Slice_Skew"
                    ] = var_slice_skew

                    del scan, metadata
                    gc.collect()
            except Exception as e:
                print(
                    f"DICOM pixel feature extraction failed; falling back to header-only metadata. Reason: {repr(e)}"
                )
                self.train = self._add_empty_image_feature_columns(self.train)
                self.test = self._add_empty_image_feature_columns(self.test)
                for dataset, df in [("train", self.train), ("test", self.test)]:
                    for patient_name in df["Patient"].unique():
                        try:
                            md = self.load_scan_metadata_only(dataset, patient_name)
                            voxel_vol = (
                                md["SliceSpacing"]
                                * md["PixelSpacing"][0]
                                * md["PixelSpacing"][1]
                            )
                            df.loc[df["Patient"] == patient_name, "Scan_Var"] = (
                                voxel_vol
                            )
                            df.loc[df["Patient"] == patient_name, "Scan_Std"] = np.sqrt(
                                max(voxel_vol, 0.0)
                            )
                            df.loc[df["Patient"] == patient_name, "Scan_Mean"] = (
                                voxel_vol
                            )
                        except Exception:
                            continue

                for c in [
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
                    self.train[c] = self.train[c].astype(np.float32)
                    self.test[c] = self.test[c].astype(np.float32)

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
                    scaler = StandardScaler()
                    scaler.fit(self.train.loc[:, scale_features])
                    self.train.loc[:, scale_features] = scaler.transform(
                        self.train.loc[:, scale_features]
                    )
                    self.test.loc[:, scale_features] = scaler.transform(
                        self.test.loc[:, scale_features]
                    )

                return self.train.copy(deep=True), self.test.copy(deep=True)

        print(f'\nCreating Image Features for Test Set\n{"-" * 36}')
        try:
            uniq_test_patients = self.test["Patient"].unique()
            for i, patient_name in enumerate(uniq_test_patients):
                scan, metadata = self.load_scan("test", patient_name)
                scan_size = scan.nbytes >> 20
                print(
                    f"[{i + 1}/{len(uniq_test_patients)}] {patient_name} Shape: {scan.shape} - Size: {scan_size} MB"
                )

                volume = (
                    (metadata["SliceSpacing"] * scan.shape[0])
                    * (metadata["PixelSpacing"][0] * scan.shape[1])
                    * (metadata["PixelSpacing"][1] * scan.shape[2])
                )
                voxel_vol = volume / (scan.shape[0] * scan.shape[1] * scan.shape[2])
                self.test.loc[self.test["Patient"] == patient_name, "VoxelVolume"] = (
                    voxel_vol
                )

                (
                    scan_skew,
                    scan_kurt,
                    scan_mean,
                    scan_std,
                    scan_var,
                    std_slice_skew,
                    var_slice_skew,
                ) = self._scan_stats(scan)
                self.test.loc[self.test["Patient"] == patient_name, "Scan_Skew"] = (
                    scan_skew
                )
                self.test.loc[self.test["Patient"] == patient_name, "Scan_Kurtosis"] = (
                    scan_kurt
                )
                self.test.loc[self.test["Patient"] == patient_name, "Scan_Mean"] = (
                    scan_mean
                )
                self.test.loc[self.test["Patient"] == patient_name, "Scan_Std"] = (
                    scan_std
                )
                self.test.loc[self.test["Patient"] == patient_name, "Scan_Var"] = (
                    scan_var
                )

                self.test.loc[
                    self.test["Patient"] == patient_name, "Scan_Min_Volume"
                ] = (float((scan == self.y_min).sum()) * voxel_vol)
                self.test.loc[
                    self.test["Patient"] == patient_name, "Scan_Max_Volume"
                ] = (float((scan == self.y_max).sum()) * voxel_vol)

                self.test.loc[
                    self.test["Patient"] == patient_name, "Std_Slice_Skew"
                ] = std_slice_skew
                self.test.loc[
                    self.test["Patient"] == patient_name, "Var_Slice_Skew"
                ] = var_slice_skew

                del scan, metadata
                gc.collect()
        except Exception as e:
            print(
                f"Test DICOM feature extraction failed; using empty image features. Reason: {repr(e)}"
            )
            self.train = self._add_empty_image_feature_columns(self.train)
            self.test = self._add_empty_image_feature_columns(self.test)

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
            self.train = self._add_empty_image_feature_columns(self.train)
            self.test = self._add_empty_image_feature_columns(self.test)

            scaler = StandardScaler()
            scaler.fit(self.train.loc[:, scale_features])

            self.train.loc[:, scale_features] = scaler.transform(
                self.train.loc[:, scale_features]
            )
            self.test.loc[:, scale_features] = scaler.transform(
                self.test.loc[:, scale_features]
            )

        if "VoxelVolume" in self.test.columns:
            self.test_out = self.test.drop(columns=["VoxelVolume"]).copy(deep=True)
        else:
            self.test_out = self.test.copy(deep=True)

        return self.train.copy(deep=True), self.test_out




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
        y_true = K.cast(y_true, "float32")
        y_pred = K.cast(y_pred, "float32")
        y_true = K.reshape(y_true, (-1, 1))

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
        y_true = K.cast(y_true, "float32")
        y_true = K.reshape(y_true, (-1, 1))
        quantiles = K.constant(
            np.array([self.qr_parameters["quantiles"]]), dtype="float32"
        )
        error = y_true - y_pred
        return K.mean(K.maximum(quantiles * error, (quantiles - 1) * error))

    def get_model(self, input_shape, m):
        if m == "MLP":
            input_layer = Input(shape=(input_shape,))
            x = Dense(2**7, activation="relu")(input_layer)
            x = Dropout(0.01)(x)
            x = Dense(2**7, activation="relu")(x)
            x = Dropout(0.01)(x)
            output_layer = Dense(2, activation="relu")(x)

            model = Model(input_layer, output_layer)
            model.compile(
                loss=self.laplace_log_likelihood_loss,
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.mlp_parameters["lr"]
                ),
                metrics=[self.laplace_log_likelihood_loss],
            )
            return model

        if m == "QR":
            input_layer = Input(shape=(input_shape,))
            x = Dense(2**7, activation="relu")(input_layer)
            x = GaussianDropout(0.01)(x)
            x = Dense(2**7, activation="relu")(x)
            x = GaussianDropout(0.01)(x)
            p1 = Dense(3, activation="linear")(x)
            p2 = Dense(3, activation="relu")(x)
            output_layer = Lambda(lambda z: z[0] + K.cumsum(z[1], axis=1))([p1, p2])

            model = Model(input_layer, output_layer)
            model.compile(
                loss=self.tilted_loss,
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.qr_parameters["lr"]
                ),
                metrics=[self.laplace_log_likelihood_loss],
            )
            return model

        raise ValueError(f"Unknown model type: {m}")

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

            for cv in self.cv:
                for fold in sorted(X_train[f"CV{cv}_Fold"].unique()):

                    trn_idx = X_train.loc[X_train[f"CV{cv}_Fold"] != fold].index
                    val_idx = X_train.loc[X_train[f"CV{cv}_Fold"] == fold].index

                    X_trn = X_train.loc[trn_idx, self.predictors].to_numpy(
                        dtype=np.float32, copy=False
                    )
                    y_trn = y_train.loc[trn_idx].to_numpy(dtype=np.float32, copy=False)
                    X_val = X_train.loc[val_idx, self.predictors].to_numpy(
                        dtype=np.float32, copy=False
                    )
                    y_val = y_train.loc[val_idx].to_numpy(dtype=np.float32, copy=False)

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
                        tmp = (
                            df_train_ref.loc[val_idx]
                            .groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index()
                        )
                        for df_patient in np.array_split(
                            tmp, df_train_ref.loc[val_idx, "Patient"].nunique()
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
                        tmp = (
                            df_train_ref.loc[val_idx]
                            .groupby("Patient")
                            .nth([-1, -2, -3])
                            .reset_index()
                        )
                        for df_patient in np.array_split(
                            tmp, df_train_ref.loc[val_idx, "Patient"].nunique()
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
                    tmp = (
                        df_train_ref.groupby("Patient").nth([-1, -2, -3]).reset_index()
                    )
                    for df_patient in np.array_split(
                        tmp, df_train_ref["Patient"].nunique()
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
                    tmp = (
                        df_train_ref.groupby("Patient").nth([-1, -2, -3]).reset_index()
                    )
                    for df_patient in np.array_split(
                        tmp, df_train_ref["Patient"].nunique()
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
                    f'{"-" * 30}\nCV {cv} {m} All Measurement OOF Score {oof_all_score:.6} - Final 3 Measurement OOF Score {np.mean(oof_final_scores):.6} [Std: {np.std(oof_final_scores):.6}]\n{"-" * 30}\n'
                )

    def predict(self, X_test):
        Xmat = X_test[self.predictors].to_numpy(dtype=np.float32, copy=False)

        for cv in self.cv:
            mlp_predictions = np.zeros((len(X_test), 2), dtype=np.float32)
            if len(self.mlp_models[f"CV{cv}"]) == 0:
                raise RuntimeError(f"No trained MLP models found for CV{cv}")
            denom = float(len(self.mlp_models[f"CV{cv}"]))
            for model in self.mlp_models[f"CV{cv}"]:
                mlp_predictions += (
                    model.predict(Xmat, verbose=0).astype(np.float32, copy=False)
                    / denom
                )

            X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_predictions[:, 0]
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_predictions[:, 1]

            qr_predictions = np.zeros(
                (len(X_test), len(self.qr_parameters["quantiles"])), dtype=np.float32
            )
            if len(self.qr_models[f"CV{cv}"]) == 0:
                raise RuntimeError(f"No trained QR models found for CV{cv}")
            denom = float(len(self.qr_models[f"CV{cv}"]))
            for model in self.qr_models[f"CV{cv}"]:
                qr_predictions += (
                    model.predict(Xmat, verbose=0).astype(np.float32, copy=False)
                    / denom
                )

            for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_predictions[:, i]

    def plot_predictions(self, df, patient):
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

        cols = ["Weeks"]
        if "FVC" in df.columns:
            cols += ["FVC"]
        if all(c in df.columns for c in mlp_prediction_columns[::2]):
            cols += mlp_prediction_columns[::2]
        if all(c in df.columns for c in qr_prediction_columns[1::3]):
            cols += qr_prediction_columns[1::3]

        ax = df[cols].set_index("Weeks").plot(figsize=(30, 6))
        if all(c in df.columns for c in qr_prediction_columns[:3]):
            ax.fill_between(
                df["Weeks"],
                df[qr_prediction_columns[2]],
                df[qr_prediction_columns[0]],
                alpha=0.1,
                label="CV1 QR Prediction Interval",
                color="red",
            )

        ax.set_xlabel("")
        ax.set_ylabel("")
        if mlp_scores is not None or qr_scores is not None:
            ax.set_title(
                f"Patient: {patient} - MLP Scores: {mlp_scores} QR Scores: {qr_scores}",
                size=25,
                pad=25,
            )
        else:
            ax.set_title(f"Patient: {patient}", size=25, pad=25)
        ax.legend()
        plt.show()




## === cell 7
seed_everything(SEED)

X_train = df_train.drop(columns=["FVC"])
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
    "mlp_parameters": {"lr": 0.00025, "epochs": 800, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.0005,
        "epochs": 100,
        "batch_size": 2**5,
    },
}

missing = [c for c in model_parameters["predictors"] if c not in X_train.columns]
if missing:
    raise KeyError(f"Missing predictors in training data: {missing}")

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train, df_train_ref=df_train)
qr_mlp.predict(df_test)

print(
    "Test prediction columns present:",
    [c for c in df_test.columns if c.startswith("CV1_")][:10],
)




## === cell 8
if False:
    for patient, dfp in list(df_train.groupby("Patient"))[:3]:
        try:
            qr_mlp.plot_predictions(dfp, patient)
        except Exception as e:
            print(f"Plot failed for {patient}: {repr(e)}")
            break




## === cell 9
class SubmissionPipeline:

    def __init__(self, df_train, df_test, sample_submission):
        self.df_train = df_train
        self.df_test = df_test
        self.sample_submission = sample_submission.copy(deep=True)

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
            train_fvc_col = f"{model}_FVC_Predictions"
            train_conf_col = f"{model}_Confidence_Predictions"

            if (
                train_fvc_col in self.df_train.columns
                and train_conf_col in self.df_train.columns
            ):
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[train_fvc_col],
                    self.df_train[train_conf_col],
                )
                print(f"Single Model {model} Score: {score:.6}")
            else:
                print(
                    f"Train OOF columns for {model} not found; skipping OOF score print."
                )

            if (
                train_fvc_col not in self.df_test.columns
                or train_conf_col not in self.df_test.columns
            ):
                raise ValueError(
                    f"Missing test prediction columns: {train_fvc_col}, {train_conf_col}"
                )

            self.df_test["FVC"] = self.df_test[train_fvc_col]
            self.df_test["Confidence"] = self.df_test[train_conf_col]

        elif model.split("_")[1] == "QR":
            q = [0.25, 0.5, 0.75]
            mid = f"{model}_QR_{q[1]}_Predictions"
            lo = f"{model}_QR_{q[0]}_Predictions"
            hi = f"{model}_QR_{q[2]}_Predictions"

            if all(c in self.df_train.columns for c in [mid, lo, hi]):
                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[mid],
                    (self.df_train[hi] - self.df_train[lo]),
                )
                print(f"Single Model {model} Score: {score:.6}")
            else:
                print(
                    f"Train OOF columns for {model} not found; skipping OOF score print."
                )

            if not all(c in self.df_test.columns for c in [mid, lo, hi]):
                raise ValueError(f"Missing test prediction columns for {model} QR")

            self.df_test["FVC"] = self.df_test[mid]
            self.df_test["Confidence"] = self.df_test[hi] - self.df_test[lo]
        else:
            raise ValueError(f"Unknown model key: {model}")

        self.df_test["Confidence"] = self.df_test["Confidence"].astype(np.float32)
        self.df_test["Confidence"] = np.maximum(self.df_test["Confidence"].values, 70.0)

        out = self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)
        out = self.sample_submission[["Patient_Week"]].merge(
            out, on="Patient_Week", how="left"
        )

        if out["FVC"].isna().any():
            out["FVC"] = out["FVC"].fillna(self.df_train["FVC"].median())
        if out["Confidence"].isna().any():
            out["Confidence"] = out["Confidence"].fillna(250.0)
        out["Confidence"] = np.maximum(
            out["Confidence"].values.astype(np.float32), 70.0
        )

        print(f'\n{out[["FVC", "Confidence"]].describe()}')
        return out




## === cell 10
sub = SubmissionPipeline(df_train, df_test, sample_submission=df_submission)
df_out = sub.single_model(model="CV1_MLP")

df_out["Patient_Week"] = df_out["Patient_Week"].astype(str)
df_out["FVC"] = df_out["FVC"].astype(np.float32)
df_out["Confidence"] = df_out["Confidence"].astype(np.float32)

df_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_out.shape)
print(df_out.head())




## === cell 11
df_out

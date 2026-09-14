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

-6.933166107009767

# 6. Current score

-11.58849

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -11.58849) has done: 'I fix the environment/runtime failures first by removing incompatible imports (the protobuf-related TensorFlow crash) and by updating scikit-learn API usage (remove deprecated `normalize` in `LinearRegression`). Then I fix the fold feature plumbing: the training code expects `CV*_Fold` columns inside `X_train`, so we must not drop them. Next, I ensure image feature generation doesn’t reference a missing external features dataset and instead computes features directly from DICOMs for both train/test (keeps the same feature logic, just makes it runnable). Finally, I make the submission builder robust to column ordering and always produce a valid `submission.csv` with the exact required columns.'

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

import cv2
import pydicom

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

TF_AVAILABLE = False
try:
    import tensorflow as tf  # noqa
    import tensorflow.keras.backend as K  # noqa
    from tensorflow.keras.models import Model  # noqa
    from tensorflow.keras.layers import Input, Dense, Dropout, GaussianDropout  # noqa

    TF_AVAILABLE = True
except Exception as e:
    print(
        "WARNING: TensorFlow could not be imported in this environment. Falling back to a deterministic numpy model."
    )
    print("TensorFlow import error:", repr(e))
    TF_AVAILABLE = False

SEED = 1337


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
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
print(f'Test Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f"Sample Submission Shape = {df_submission.shape}")




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
            z = (last2 - last2.mean()) / (last2.std() + 1e-8)
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
    f'Test Set (Tabular Features) Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}'
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
        base = f"../input/osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}"
        files = os.listdir(base)
        patient_directory = [pydicom.dcmread(f"{base}/{s}") for s in files]

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

        patient_directory = list(
            np.array(patient_directory, dtype=object)[non_duplicate_idx]
        )

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
                metadata["SliceSpacing"] = float(mode(diffs, keepdims=True).mode[0])

        scan = np.zeros(
            (len(patient_directory), self.resize_shape[0], self.resize_shape[1]),
            dtype=np.int16,
        )
        for i, s in enumerate(patient_directory):
            s_processed = self.crop(s.pixel_array)
            s_processed = self.resize(s_processed)
            s_processed = self.window(
                s_processed,
                float(s.RescaleSlope),
                float(s.RescaleIntercept),
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

    def _add_features_for_patient(self, df, dataset, patient_name):
        scan, metadata = self.load_scan(dataset, patient_name)
        if scan.size == 0:
            df.loc[df["Patient"] == patient_name, "VoxelVolume"] = 0.0
            df.loc[df["Patient"] == patient_name, "Scan_Skew"] = 0.0
            df.loc[df["Patient"] == patient_name, "Scan_Kurtosis"] = 0.0
            df.loc[df["Patient"] == patient_name, "Scan_Mean"] = 0.0
            df.loc[df["Patient"] == patient_name, "Scan_Std"] = 0.0
            df.loc[df["Patient"] == patient_name, "Scan_Var"] = 0.0
            df.loc[df["Patient"] == patient_name, "Scan_Min_Volume"] = 0.0
            df.loc[df["Patient"] == patient_name, "Scan_Max_Volume"] = 0.0
            df.loc[df["Patient"] == patient_name, "Std_Slice_Skew"] = 0.0
            df.loc[df["Patient"] == patient_name, "Var_Slice_Skew"] = 0.0
            return

        voxel_total_volume = (
            (metadata["SliceSpacing"] * scan.shape[0])
            * (metadata["PixelSpacing"][0] * scan.shape[1])
            * (metadata["PixelSpacing"][1] * scan.shape[2])
        )
        voxel_volume = voxel_total_volume / (
            scan.shape[0] * scan.shape[1] * scan.shape[2]
        )
        df.loc[df["Patient"] == patient_name, "VoxelVolume"] = voxel_volume

        flat = scan.flatten().astype(np.float32)
        df.loc[df["Patient"] == patient_name, "Scan_Skew"] = float(skew(flat))
        df.loc[df["Patient"] == patient_name, "Scan_Kurtosis"] = float(kurtosis(flat))
        df.loc[df["Patient"] == patient_name, "Scan_Mean"] = float(flat.mean())
        df.loc[df["Patient"] == patient_name, "Scan_Std"] = float(flat.std())
        df.loc[df["Patient"] == patient_name, "Scan_Var"] = float(flat.var())
        df.loc[df["Patient"] == patient_name, "Scan_Min_Volume"] = float(
            (scan == self.y_min).sum() * voxel_volume
        )
        df.loc[df["Patient"] == patient_name, "Scan_Max_Volume"] = float(
            (scan == self.y_max).sum() * voxel_volume
        )

        slice_skews = [skew(s.flatten().astype(np.float32)) for s in scan]
        df.loc[df["Patient"] == patient_name, "Std_Slice_Skew"] = float(
            np.std(slice_skews)
        )
        df.loc[df["Patient"] == patient_name, "Var_Slice_Skew"] = float(
            np.var(slice_skews)
        )

        del scan, metadata, flat, slice_skews
        gc.collect()

    def create_image_features(self):
        print(f'Creating Image Features for Training Set\n{"-" * 40}')
        for i, patient_name in enumerate(self.train["Patient"].unique()):
            if i % 20 == 0:
                print(f'  processed {i}/{self.train["Patient"].nunique()} patients...')
            self._add_features_for_patient(self.train, "train", patient_name)

        print(f'\nCreating Image Features for Test Set\n{"-" * 36}')
        for i, patient_name in enumerate(self.test["Patient"].unique()):
            print(f'[{i + 1}/{self.test["Patient"].nunique()}] {patient_name}')
            self._add_features_for_patient(self.test, "test", patient_name)

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
    f'Test Set (Image + Tabular Features) Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}'
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1284926112.py in <cell line: 0>()
     10 )
     11 
---> 12 df_train, df_test = image_data_preprocessor.create_image_features()
     13 
     14 print(

/tmp/ipykernel_11/338434875.py in create_image_features(self)
    198             if i % 20 == 0:
    199                 print(f'  processed {i}/{self.train["Patient"].nunique()} patients...')
--> 200             self._add_features_for_patient(self.train, "train", patient_name)
    201 
    202         print(f'\nCreating Image Features for Test Set\n{"-" * 36}')

/tmp/ipykernel_11/338434875.py in _add_features_for_patient(self, df, dataset, patient_name)
    144 
    145     def _add_features_for_patient(self, df, dataset, patient_name):
--> 146         scan, metadata = self.load_scan(dataset, patient_name)
    147         if scan.size == 0:
    148             # Safe defaults if scan fails

/tmp/ipykernel_11/338434875.py in load_scan(self, dataset, patient_name)
     86 
     87         patient_directory = list(
---> 88             np.array(patient_directory, dtype=object)[non_duplicate_idx]
     89         )
     90 

TypeError: Dataset.__array__() takes 1 positional argument but 2 were given

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
        return float(np.mean(score))

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
        if not TF_AVAILABLE:
            return None

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
            output_layer = Dense(3, activation="relu")(x)
            model = Model(input_layer, output_layer)
            model.compile(
                loss=self.tilted_loss,
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.qr_parameters["lr"]
                ),
                metrics=[self.laplace_log_likelihood_loss],
            )
            return model

        raise ValueError("Unknown model type")

    def _fit_ridge(self, X, y, l2=1e-6):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        A = Xb.T @ Xb + l2 * np.eye(Xb.shape[1])
        w = np.linalg.solve(A, Xb.T @ y)
        return w  # (p+1,1)

    def _predict_lin(self, X, w):
        X = np.asarray(X, dtype=np.float64)
        Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return (Xb @ w).reshape(-1)

    def train(self, X_train, y_train):
        self.mlp_models = {"CV1": [], "CV2": [], "CV3": []}
        self.qr_models = {"CV1": [], "CV2": [], "CV3": []}

        for cv in self.cv:
            fold_col = f"CV{cv}_Fold"
            if fold_col not in X_train.columns:
                raise KeyError(
                    f"Missing {fold_col} in X_train. Do not drop CV fold columns before training."
                )

        models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
        for m in models:
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + (len(m)))}')
            for cv in self.cv:
                for fold in sorted(X_train[f"CV{cv}_Fold"].unique()):
                    trn_idx = X_train.loc[X_train[f"CV{cv}_Fold"] != fold].index
                    val_idx = X_train.loc[X_train[f"CV{cv}_Fold"] == fold].index

                    X_trn = X_train.loc[trn_idx, self.predictors]
                    y_trn = y_train.loc[trn_idx]
                    X_val = X_train.loc[val_idx, self.predictors]
                    y_val = y_train.loc[val_idx]

                    if TF_AVAILABLE:
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
                            predictions = model.predict(X_val, verbose=0)
                            oof_predictions = predictions[:, 0]
                            oof_confidence = predictions[:, 1]
                        else:
                            model.fit(
                                X_trn,
                                y_trn,
                                epochs=self.qr_parameters["epochs"],
                                batch_size=self.qr_parameters["batch_size"],
                                verbose=0,
                            )
                            self.qr_models[f"CV{cv}"].append(model)
                            predictions = model.predict(X_val, verbose=0)
                            oof_predictions = predictions[:, 1]
                            oof_confidence = predictions[:, 2] - predictions[:, 0]
                    else:
                        w = self._fit_ridge(X_trn.values, y_trn.values)
                        pred = self._predict_lin(X_val.values, w)
                        resid = y_trn.values - self._predict_lin(X_trn.values, w)
                        sigma = np.full_like(
                            pred, max(70.0, float(np.std(resid) + 1e-6))
                        )
                        if m == "MLP":
                            oof_predictions = pred
                            oof_confidence = sigma
                            self.mlp_models[f"CV{cv}"].append(
                                (w, float(np.std(resid) + 1e-6))
                            )
                        else:
                            oof_predictions = pred
                            oof_confidence = 2.0 * sigma
                            self.qr_models[f"CV{cv}"].append(
                                (w, float(np.std(resid) + 1e-6))
                            )

                    if m == "MLP":
                        df_train.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = (
                            oof_predictions
                        )
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
                    else:
                        q = self.qr_parameters["quantiles"]
                        if TF_AVAILABLE:
                            df_train.loc[val_idx, f"CV{cv}_QR_{q[0]}_Predictions"] = (
                                predictions[:, 0]
                            )
                            df_train.loc[val_idx, f"CV{cv}_QR_{q[1]}_Predictions"] = (
                                predictions[:, 1]
                            )
                            df_train.loc[val_idx, f"CV{cv}_QR_{q[2]}_Predictions"] = (
                                predictions[:, 2]
                            )
                        else:
                            s = oof_confidence / 2.0
                            df_train.loc[val_idx, f"CV{cv}_QR_{q[0]}_Predictions"] = (
                                oof_predictions - s
                            )
                            df_train.loc[val_idx, f"CV{cv}_QR_{q[1]}_Predictions"] = (
                                oof_predictions
                            )
                            df_train.loc[val_idx, f"CV{cv}_QR_{q[2]}_Predictions"] = (
                                oof_predictions + s
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
                        f"CV {cv} {m} Fold {int(fold)} - X_train: {X_trn.shape} X_val: {X_val.shape} - All Measurement Score: {oof_score_all:.6} - Final 3 Measurement Score {np.mean(fold_final_scores):.6}"
                    )

    def predict(self, X_test):
        for cv in self.cv:
            if TF_AVAILABLE:
                mlp_predictions = np.zeros((len(X_test), 2), dtype=np.float32)
                for model in self.mlp_models[f"CV{cv}"]:
                    mlp_predictions += model.predict(
                        X_test[self.predictors], verbose=0
                    ) / len(self.mlp_models[f"CV{cv}"])
                X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_predictions[:, 0]
                X_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_predictions[:, 1]

                qr_predictions = np.zeros(
                    (len(X_test), len(self.qr_parameters["quantiles"])),
                    dtype=np.float32,
                )
                for model in self.qr_models[f"CV{cv}"]:
                    qr_predictions += model.predict(
                        X_test[self.predictors], verbose=0
                    ) / len(self.qr_models[f"CV{cv}"])
                for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                    X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_predictions[:, i]
            else:
                if len(self.mlp_models[f"CV{cv}"]) > 0:
                    preds = []
                    sigmas = []
                    for w, s in self.mlp_models[f"CV{cv}"]:
                        preds.append(
                            self._predict_lin(X_test[self.predictors].values, w)
                        )
                        sigmas.append(s)
                    mean_pred = np.mean(np.vstack(preds), axis=0)
                    sigma = max(70.0, float(np.mean(sigmas)))
                    X_test[f"CV{cv}_MLP_FVC_Predictions"] = mean_pred
                    X_test[f"CV{cv}_MLP_Confidence_Predictions"] = sigma

                if len(self.qr_models[f"CV{cv}"]) > 0:
                    preds = []
                    sigmas = []
                    for w, s in self.qr_models[f"CV{cv}"]:
                        preds.append(
                            self._predict_lin(X_test[self.predictors].values, w)
                        )
                        sigmas.append(s)
                    mean_pred = np.mean(np.vstack(preds), axis=0)
                    s = max(70.0, float(np.mean(sigmas)))
                    q = self.qr_parameters["quantiles"]
                    X_test[f"CV{cv}_QR_{q[0]}_Predictions"] = mean_pred - s
                    X_test[f"CV{cv}_QR_{q[1]}_Predictions"] = mean_pred
                    X_test[f"CV{cv}_QR_{q[2]}_Predictions"] = mean_pred + s

    def plot_predictions(self, df, patient):
        return




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
    "mlp_parameters": {"lr": 0.0005, "epochs": 200, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.0005,
        "epochs": 750,
        "batch_size": 2**5,
    },
}

qr_mlp = QuantileRegressorMLP(**model_parameters)
qr_mlp.train(X_train, y_train)
qr_mlp.predict(df_test)




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
            print(f"Single Model {model} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[fvc_col]
            self.df_test["Confidence"] = self.df_test[conf_col]

        elif model.split("_")[1] == "QR":
            q = [0.25, 0.5, 0.75]
            q0, q1, q2 = (
                f"{model}_{q[0]}_Predictions",
                f"{model}_{q[1]}_Predictions",
                f"{model}_{q[2]}_Predictions",
            )
            for c in (q0, q1, q2):
                if c not in self.df_train.columns:
                    raise KeyError(f"Missing required column for {model}: {c}")
            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[q1],
                (self.df_train[q2] - self.df_train[q0]),
            )
            print(f"Single Model {model} Score: {score:.6}")
            self.df_test["FVC"] = self.df_test[q1]
            self.df_test["Confidence"] = self.df_test[q2] - self.df_test[q0]
        else:
            raise ValueError("model must be like 'CV1_MLP' or 'CV1_QR'")

        self.df_test["Confidence"] = np.maximum(
            self.df_test["Confidence"].astype(np.float32), 70.0
        )
        self.df_test["FVC"] = self.df_test["FVC"].astype(np.float32)

        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 9
sub = SubmissionPipeline(df_train, df_test)

df_sub = sub.single_model(model="CV1_QR")

sample = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
df_sub = sample[["Patient_Week"]].merge(df_sub, on="Patient_Week", how="left")

df_sub["FVC"] = df_sub["FVC"].fillna(df_sub["FVC"].median()).astype(np.float32)
df_sub["Confidence"] = df_sub["Confidence"].fillna(200.0).astype(np.float32)

df_sub.to_csv("submission.csv", index=False)
print(df_sub.head())
print("Wrote submission.csv with shape:", df_sub.shape)

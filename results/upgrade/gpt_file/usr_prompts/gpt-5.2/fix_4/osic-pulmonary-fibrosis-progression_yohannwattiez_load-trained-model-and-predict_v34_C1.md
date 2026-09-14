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

-6.988687156750365

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pathlib
import time

import pydicom

from tensorflow import keras as K
from tensorflow.keras import layers as L

from scipy.ndimage import zoom
import scipy.ndimage as ndimage
from skimage import measure, segmentation

from math import ceil

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
raw_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
X_prediction = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 2
TEST_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression/test"

DESIRED_SIZE = (30, 256, 256)
BATCH_SIZE = 256

clip_bounds = (-1000, 200)
pre_calculated_mean = 0.02865046213070556



## === cell 3
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

rename_cols = {
    "Weeks_y": "Base_week",
    "Weeks_x": "Weeks",
    "Percent": "Base_percent",
    "FVC": "Base_FVC",
}
X_prediction = (
    X_prediction.merge(raw_test, how="left", left_on="Patient", right_on="Patient")
    .rename(columns=rename_cols)[
        [
            "Patient",
            "Base_week",
            "Base_FVC",
            "Base_percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ]
    ]
    .reset_index(drop=True)
)



## === cell 4
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index="", name="", **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)
        return d_out

    def fit_transform(self, X, categories, index, name, **kwargs):
        self.fit(X)
        return self.transform(X, categories=categories, index=index, name=name)

    def get_new_columns(self, X, name, categories):
        new_columns = []
        for j in range(len(categories)):
            new_columns.append("{}_{}".format(name, categories[j]))
        return new_columns


from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


class data_preparation:
    def __init__(self):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()

    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)
        try:
            data["Sex"] = self.enc_sex.transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ).astype(int),
                ],
                axis=1,
            )
        except NotFittedError:
            data["Sex"] = self.enc_sex.fit_transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.fit_transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.fit_transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ).astype(int),
                ],
                axis=1,
            )
        return data




## === cell 5
data_prep = data_preparation()

X_prediction = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)

expected_smoke_cols = ["_Currently smokes", "_Ex-smoker", "_Never smoked"]
for c in expected_smoke_cols:
    if c not in X_prediction.columns:
        X_prediction[c] = 0

for c in [
    "Weeks",
    "Base_week",
    "Base_FVC",
    "Base_percent",
    "Age",
    "Sex",
] + expected_smoke_cols:
    X_prediction[c] = pd.to_numeric(X_prediction[c], errors="coerce")

X_prediction[["Base_week"]] = X_prediction[["Base_week"]].fillna(0)
X_prediction[["Base_FVC", "Base_percent", "Age", "Sex"]] = (
    X_prediction[["Base_FVC", "Base_percent", "Age", "Sex"]]
    .fillna(method="ffill")
    .fillna(method="bfill")
    .fillna(0)
)
X_prediction[expected_smoke_cols] = (
    X_prediction[expected_smoke_cols].fillna(0).astype(int)
)




## === cell 6
class ConvertToHU:
    def __call__(self, imgs, dicom):
        intercept = dicom.RescaleIntercept
        slope = dicom.RescaleSlope
        imgs = (np.array(imgs.to_list()) * slope + intercept).astype(np.int16)
        return imgs


convertohu = ConvertToHU()




## === cell 7
class Clip:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, image):
        image = image.copy()
        image[image < self.min] = self.min
        image[image > self.max] = self.max
        return image


clip = Clip(clip_bounds)




## === cell 8
class MaskWatershed:
    def __init__(self, min_hu, iterations):
        self.min_hu = min_hu
        self.iterations = iterations

    def __call__(self, image, dicom):
        stack = []
        for slice_idx in range(image.shape[0]):
            sliced = image[slice_idx]
            stack.append(self.seperate_lungs(sliced, self.min_hu, self.iterations))
        return np.stack(stack)

    @staticmethod
    def seperate_lungs(image, min_hu, iterations):
        h, w = image.shape[0], image.shape[1]

        marker_internal, marker_external, marker_watershed = (
            MaskWatershed.generate_markers(image)
        )

        sobel_filtered_dx = ndimage.sobel(image, 1)
        sobel_filtered_dy = ndimage.sobel(image, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        m = np.max(sobel_gradient)
        if m > 0:
            sobel_gradient *= 255.0 / m

        watershed = segmentation.watershed(sobel_gradient, marker_watershed)

        outline = ndimage.morphological_gradient(watershed, size=(3, 3))
        outline = outline.astype(bool)

        blackhat_struct = [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ]

        blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)

        outline = outline + ndimage.black_tophat(outline, structure=blackhat_struct)

        lungfilter = np.bitwise_or(marker_internal, outline)
        lungfilter = ndimage.binary_closing(
            lungfilter, structure=np.ones((5, 5)), iterations=3
        )

        segmented = np.where(lungfilter == 1, image, min_hu * np.ones((h, w)))
        return segmented

    @staticmethod
    def generate_markers(image, threshold=-400):
        h, w = image.shape[0], image.shape[1]

        marker_internal = image < threshold
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)

        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()

        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0

        marker_internal = marker_internal_labels > 0

        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a

        marker_watershed = np.zeros((h, w), dtype=np.int32)
        marker_watershed += marker_internal.astype(np.int32) * 255
        marker_watershed += marker_external.astype(np.int32) * 128

        return marker_internal, marker_external, marker_watershed


maskwatershed = MaskWatershed(min_hu=min(clip_bounds), iterations=1)




## === cell 9
class Normalize:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, image):
        image = image.astype(np.float32)
        image = (image - self.min) / (self.max - self.min)
        return image


class ZeroCenter:
    def __init__(self, pre_calculated_mean):
        self.pre_calculated_mean = pre_calculated_mean

    def __call__(self, image):
        return image - self.pre_calculated_mean


normalize = Normalize(bounds=clip_bounds)
zerocenter = ZeroCenter(pre_calculated_mean=pre_calculated_mean)




## === cell 10
def sort_function(x):  # Get the files in the right order
    return int(x.split(".")[0])


def _read(path, patients=[], desired_size=(60, 512, 512)):
    X = np.empty(
        np.concatenate(([len(patients), 1], np.array(desired_size))), dtype=np.float32
    )
    i = 0
    for patient in patients:
        patient_dir = os.path.join(path, patient)
        list_patient_files = sorted(
            [fn for fn in os.listdir(patient_dir) if fn.endswith(".dcm")],
            key=sort_function,
        )
        if len(list_patient_files) == 0:
            raise FileNotFoundError(f"No DICOM files found for patient: {patient_dir}")

        dicom0 = pydicom.dcmread(os.path.join(patient_dir, list_patient_files[0]))

        filepaths = [os.path.join(patient_dir, fn) for fn in list_patient_files]
        slices = [pydicom.dcmread(fp).pixel_array for fp in filepaths]

        df = convertohu(pd.Series(slices), dicom0)
        df = zoom(df, np.array(desired_size) / np.array(df.shape), mode="nearest")

        X[i, 0, :, :, :] = zerocenter(normalize(maskwatershed(clip(df), dicom0)))
        i += 1
    return X




## === cell 11
temp_SELECTED_COLUMNS = [
    "Weeks",
    "Base_week",
    "Base_FVC",
    "Base_percent",
    "Age",
    "Sex",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]


class DataGenerator(K.utils.Sequence):
    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __init__(
        self,
        train,
        list_IDs,
        batch_size=1,
        desired_size=(10, 512, 512),
        img_path=TEST_PATH,
        *args,
        **kwargs,
    ):
        self.train = train
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.desired_size = desired_size
        self.img_path = img_path
        self.on_epoch_end()

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        patients = self.train.loc[list_IDs_temp, "Patient"].unique()
        imgs = _read(self.img_path, patients=patients, desired_size=self.desired_size)

        return self.train.loc[list_IDs_temp, :].reset_index(drop=True), np.transpose(
            imgs, (0, 2, 3, 4, 1)
        )




## === cell 12
pred_SELECTED_COLUMNS = [
    "Weeks",
    "Base_week",
    "Patient",
    "Base_FVC",
    "Base_percent",
    "Age",
    "Sex",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]
pred_generator = DataGenerator(
    X_prediction[pred_SELECTED_COLUMNS],
    X_prediction.index.tolist(),
    batch_size=BATCH_SIZE,
    desired_size=DESIRED_SIZE,
    img_path=TEST_PATH,
)



## === cell 13

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

raw_train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

baseline = raw_train.loc[
    raw_train.groupby("Patient")["Weeks"].apply(lambda s: (s - 0).abs().idxmin())
].rename(columns={"Weeks": "Base_week", "FVC": "Base_FVC", "Percent": "Base_percent"})[
    ["Patient", "Base_week", "Base_FVC", "Base_percent", "Age", "Sex", "SmokingStatus"]
]

train_df = raw_train.merge(baseline, on="Patient", how="left", suffixes=("", "_base"))
train_df["Patient_Week"] = (
    train_df["Patient"].astype(str) + "_" + train_df["Weeks"].astype(str)
)

prep = data_preparation()
combined = pd.concat(
    [
        train_df[
            [
                "Patient",
                "Base_week",
                "Base_FVC",
                "Base_percent",
                "Age",
                "Sex",
                "SmokingStatus",
                "Weeks",
                "Patient_Week",
            ]
        ],
        X_prediction[
            [
                "Patient",
                "Base_week",
                "Base_FVC",
                "Base_percent",
                "Age",
                "Sex",
                "SmokingStatus",
                "Weeks",
                "Patient_Week",
            ]
        ],
    ],
    axis=0,
    ignore_index=True,
)

combined_enc = prep(combined)

train_enc = combined_enc.iloc[: len(train_df)].copy()
test_enc = combined_enc.iloc[len(train_df) :].copy()

expected_smoke_cols = ["_Currently smokes", "_Ex-smoker", "_Never smoked"]
for c in expected_smoke_cols:
    if c not in train_enc.columns:
        train_enc[c] = 0
    if c not in test_enc.columns:
        test_enc[c] = 0

feature_cols = [
    "Weeks",
    "Base_week",
    "Base_FVC",
    "Base_percent",
    "Age",
    "Sex",
] + expected_smoke_cols

X_train = (
    train_enc[feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
    .values.astype(np.float32)
)
y_train = (
    pd.to_numeric(train_df["FVC"], errors="coerce")
    .fillna(train_df["FVC"].median())
    .values.astype(np.float32)
)
groups = train_df["Patient"].values

X_test = (
    test_enc[feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
    .values.astype(np.float32)
)

gkf = GroupKFold(n_splits=5)
oof = np.zeros_like(y_train, dtype=np.float32)

for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
    model = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=SEED))]
    )
    model.fit(X_train[tr_idx], y_train[tr_idx])
    oof[va_idx] = model.predict(X_train[va_idx]).astype(np.float32)

resid = np.abs(y_train - oof)
sigma_global = float(np.maximum(70.0, np.std(y_train - oof)))

final_model = Pipeline(
    [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=SEED))]
)
final_model.fit(X_train, y_train)
fvc_pred = final_model.predict(X_test).astype(np.float32)

half_width = sigma_global
y_prediction = np.stack(
    [fvc_pred - half_width, fvc_pred, fvc_pred + half_width], axis=1
).astype(np.float32)

print(
    "Fallback clinical model used. y_prediction shape:",
    y_prediction.shape,
    "sigma_global:",
    sigma_global,
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2850149902.py in <cell line: 0>()
     40             ]
     41         ],
---> 42         X_prediction[
     43             [
     44                 "Patient",

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['SmokingStatus'] not in index"

## === cell 14
sub = pd.DataFrame(
    {
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": y_prediction[:, 1],
        "Confidence": (y_prediction[:, 2] - y_prediction[:, 0]),
    }
)

sub["Confidence"] = sub["Confidence"].astype(np.float32)
sub["Confidence"] = np.clip(sub["Confidence"], 70.0, 1e6)

sub["FVC"] = sub["FVC"].astype(np.float32)

sub = sub[["Patient_Week", "FVC", "Confidence"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1711360772.py in <cell line: 0>()
      2     {
      3         "Patient_Week": X_prediction["Patient_Week"].values,
----> 4         "FVC": y_prediction[:, 1],
      5         "Confidence": (y_prediction[:, 2] - y_prediction[:, 0]),
      6     }

NameError: name 'y_prediction' is not defined

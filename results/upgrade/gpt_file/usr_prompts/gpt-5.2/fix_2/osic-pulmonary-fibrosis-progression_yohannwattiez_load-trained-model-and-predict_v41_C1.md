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

-7.042412551619324

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
import time
import pickle
import pathlib

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L


from math import ceil

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
raw_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
raw_train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
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

rename_cols = {"Weeks_y": "Min_week", "Weeks_x": "Weeks", "FVC": "Base_FVC"}
X_prediction = (
    X_prediction.merge(raw_test, how="left", on="Patient")
    .rename(columns=rename_cols)[
        [
            "Patient",
            "Min_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ]
    ]
    .reset_index(drop=True)
)

X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]



## === cell 4
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


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
        return ["{}_{}".format(name, categories[j]) for j in range(len(categories))]


def standardisation(x, u, s):
    return (x - u) / s


def normalization(x, ma, mi):
    denom = (ma - mi) if (ma - mi) != 0 else 1.0
    return (x - mi) / denom


class data_preparation:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        self.standardisation = bool_standard
        self.normalization = bool_normalization

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

            if self.standardisation:
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
                )
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)
                data["Weeks"] = standardisation(
                    data["Weeks"], self.weeks_mean, self.weeks_std
                )

            if self.normalization:
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )
                data["Min_week"] = normalization(
                    data["Min_week"], self.min_week_max, self.min_week_min
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

            if self.standardisation:
                self.base_week_mean = data["Base_week"].mean()
                self.base_week_std = (
                    data["Base_week"].std() if data["Base_week"].std() != 0 else 1.0
                )
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )

                self.base_fvc_mean = data["Base_FVC"].mean()
                self.base_fvc_std = (
                    data["Base_FVC"].std() if data["Base_FVC"].std() != 0 else 1.0
                )
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )

                self.base_percent_mean = data["Percent"].mean()
                self.base_percent_std = (
                    data["Percent"].std() if data["Percent"].std() != 0 else 1.0
                )
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
                )

                self.age_mean = data["Age"].mean()
                self.age_std = data["Age"].std() if data["Age"].std() != 0 else 1.0
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)

                self.weeks_mean = data["Weeks"].mean()
                self.weeks_std = (
                    data["Weeks"].std() if data["Weeks"].std() != 0 else 1.0
                )
                data["Weeks"] = standardisation(
                    data["Weeks"], self.weeks_mean, self.weeks_std
                )

            if self.normalization:
                self.base_week_min = data["Base_week"].min()
                self.base_week_max = data["Base_week"].max()
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )

                self.base_fvc_min = data["Base_FVC"].min()
                self.base_fvc_max = data["Base_FVC"].max()
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )

                self.base_percent_min = data["Percent"].min()
                self.base_percent_max = data["Percent"].max()
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )

                self.age_min = data["Age"].min()
                self.age_max = data["Age"].max()
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)

                self.weeks_min = data["Weeks"].min()
                self.weeks_max = data["Weeks"].max()
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )

                self.min_week_min = data["Min_week"].min()
                self.min_week_max = data["Min_week"].max()
                data["Min_week"] = normalization(
                    data["Min_week"], self.min_week_max, self.min_week_min
                )

        return data




## === cell 5
data_prep = data_preparation(bool_normalization=True, bool_standard=False)

train_base = (
    raw_train.sort_values(["Patient", "Weeks"]).groupby("Patient").first().reset_index()
)
train_base = train_base.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
train_fit = raw_train.merge(
    train_base[["Patient", "Min_week", "Base_FVC"]], on="Patient", how="left"
)
train_fit["Base_week"] = train_fit["Weeks"] - train_fit["Min_week"]

_ = data_prep(
    train_fit[
        [
            "Patient",
            "Min_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Base_week",
        ]
    ]
)

X_prediction = (
    data_prep(X_prediction).sort_values(["Patient", "Weeks"]).reset_index(drop=True)
)



## === cell 6
from scipy.ndimage import zoom
import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation
from skimage.segmentation import watershed as sk_watershed


class ConvertToHU:
    def __call__(self, imgs, dicom):
        intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
        slope = float(getattr(dicom, "RescaleSlope", 1.0))
        imgs = (np.array(imgs.to_list()) * slope + intercept).astype(np.int16)
        return imgs


convertohu = ConvertToHU()


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
        mx = np.max(sobel_gradient)
        if mx != 0:
            sobel_gradient *= 255.0 / mx

        watershed = sk_watershed(sobel_gradient, marker_watershed)

        outline = ndimage.morphological_gradient(watershed, size=(3, 3)).astype(bool)

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

        outline = outline | ndimage.black_tophat(outline, structure=blackhat_struct)

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

        marker_watershed = np.zeros((h, w), dtype=int)
        marker_watershed += marker_internal.astype(int) * 255
        marker_watershed += marker_external.astype(int) * 128

        return marker_internal, marker_external, marker_watershed


maskwatershed = MaskWatershed(min_hu=min(clip_bounds), iterations=2)


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




## === cell 7
def sort_function(x):
    return int(x.split(".")[0])


def _read(path, patients=[], desired_size=(60, 512, 512)):
    import pydicom

    X = np.empty(
        np.concatenate(([len(patients), 1], np.array(desired_size))), dtype=np.float32
    )
    i = 0
    for patient in patients:
        patient_dir = os.path.join(path, patient)
        files = sorted(os.listdir(patient_dir), key=sort_function)
        dicom0 = pydicom.dcmread(os.path.join(patient_dir, files[0]))
        df = pd.DataFrame(files).apply(lambda x: os.path.join(patient_dir, x))
        df = convertohu(
            df.iloc[:, 0].apply(lambda x: pydicom.dcmread(x).pixel_array), dicom0
        )
        df = zoom(df, np.array(DESIRED_SIZE) / np.array(df.shape), mode="nearest")
        X[i, 0, :, :, :] = zerocenter(normalize(maskwatershed(clip(df), dicom0)))
        i += 1
    return X




## === cell 8
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




## === cell 9
pred_generator = DataGenerator(
    X_prediction,
    X_prediction.index.tolist(),
    batch_size=BATCH_SIZE,
    desired_size=DESIRED_SIZE,
    img_path=TEST_PATH,
)




## === cell 10
def make_tfsm_inference_model(savedmodel_path):
    layer = K.layers.TFSMLayer(savedmodel_path, call_endpoint="serving_default")
    inp = K.Input(shape=(None,), dtype=tf.float32)
    out = layer(inp)
    if isinstance(out, dict):
        out = list(out.values())[0]
    return K.Model(inp, out)


def make_tfsm_inference_model_5d(savedmodel_path, input_shape_5d):
    layer = K.layers.TFSMLayer(savedmodel_path, call_endpoint="serving_default")
    inp = K.Input(shape=input_shape_5d, dtype=tf.float32)
    out = layer(inp)
    if isinstance(out, dict):
        out = list(out.values())[0]
    return K.Model(inp, out)


CNN = make_tfsm_inference_model_5d(
    "/kaggle/input/3d-cnn-mlp/CNN_39",
    input_shape_5d=(DESIRED_SIZE[0], DESIRED_SIZE[1], DESIRED_SIZE[2], 1),
)
MLP = make_tfsm_inference_model("/kaggle/input/3d-cnn-mlp/model_39")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/563427237.py in <cell line: 0>()
     20 
     21 # CNN input is 5D: (D, H, W, C) with C=1 after transpose in generator
---> 22 CNN = make_tfsm_inference_model_5d(
     23     "/kaggle/input/3d-cnn-mlp/CNN_39",
     24     input_shape_5d=(DESIRED_SIZE[0], DESIRED_SIZE[1], DESIRED_SIZE[2], 1),

/tmp/ipykernel_11/563427237.py in make_tfsm_inference_model_5d(savedmodel_path, input_shape_5d)
     11 
     12 def make_tfsm_inference_model_5d(savedmodel_path, input_shape_5d):
---> 13     layer = K.layers.TFSMLayer(savedmodel_path, call_endpoint="serving_default")
     14     inp = K.Input(shape=input_shape_5d, dtype=tf.float32)
     15     out = layer(inp)

/usr/local/lib/python3.11/dist-packages/keras/src/export/tfsm_layer.py in __init__(self, filepath, call_endpoint, call_training_endpoint, trainable, name, dtype)
     64         super().__init__(trainable=trainable, name=name, dtype=dtype)
     65 
---> 66         self._reloaded_obj = tf.saved_model.load(filepath)
     67 
     68         self.filepath = filepath

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/load.py in load(export_dir, tags, options)
    910   if isinstance(export_dir, os.PathLike):
    911     export_dir = os.fspath(export_dir)
--> 912   result = load_partial(export_dir, None, tags, options)["root"]
    913   return result
    914 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/load.py in load_partial(export_dir, filters, tags, options)
   1014     tags = nest.flatten(tags)
   1015   saved_model_proto, debug_info = (
-> 1016       loader_impl.parse_saved_model_with_debug_info(export_dir))
   1017 
   1018   loader = None

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/loader_impl.py in parse_saved_model_with_debug_info(export_dir)
     57     parsed. Missing graph debug info file is fine.
     58   """
---> 59   saved_model = parse_saved_model(export_dir)
     60 
     61   debug_info_path = file_io.join(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/loader_impl.py in parse_saved_model(export_dir)
    117       raise IOError(f"Cannot parse file {path_to_pbtxt}: {str(e)}.") from e
    118   else:
--> 119     raise IOError(
    120         f"SavedModel file does not exist at: {export_dir}{os.path.sep}"
    121         f"{{{constants.SAVED_MODEL_FILENAME_PBTXT}|"

OSError: SavedModel file does not exist at: /kaggle/input/3d-cnn-mlp/CNN_39/{saved_model.pbtxt|saved_model.pb}

## === cell 11
SELECTED_COLUMNS = [
    "Weeks",
    "Percent",
    "Age",
    "Sex",
    "Min_week",
    "Base_FVC",
    "Base_week",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]

for c in SELECTED_COLUMNS:
    if c not in X_prediction.columns:
        X_prediction[c] = 0

X_prediction = X_prediction.copy()
X_prediction = X_prediction[
    ["Patient", "Weeks", "Patient_Week"]
    + [c for c in SELECTED_COLUMNS if c not in ["Weeks"]]
]




## === cell 12
def _get_first_output(t):
    if isinstance(t, dict):
        return list(t.values())[0]
    return t


y_pred_list = []
idx0 = 0

for batch_idx in range(len(pred_generator)):
    X1, X2 = pred_generator[batch_idx]

    out_imgs = _get_first_output(
        CNN(tf.convert_to_tensor(X2, dtype=tf.float32))
    )  # (n_patients, latent)

    patient_counts = X1["Patient"].value_counts()
    patients_in_counts_order = list(
        patient_counts.index
    )  # order aligns with original value_counts sorting
    patients_unique_order = list(X1["Patient"].unique())
    feat_by_patient = {
        patients_unique_order[i]: out_imgs[i] for i in range(len(patients_unique_order))
    }
    X_imgs = tf.concat(
        [
            tf.repeat(
                tf.reshape(feat_by_patient[p], (1, -1)), int(patient_counts[p]), axis=0
            )
            for p in patients_in_counts_order
        ],
        axis=0,
    )

    X_tab = tf.convert_to_tensor(np.asarray(X1[SELECTED_COLUMNS], dtype=np.float32))

    mlp_inp = tf.concat([X_imgs, X_tab], axis=1)
    out = _get_first_output(MLP(mlp_inp))
    y_pred_list.append(out.numpy())

y_prediction = np.concatenate(y_pred_list, axis=0)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2118762104.py in <cell line: 0>()
     10 
     11 for batch_idx in range(len(pred_generator)):
---> 12     X1, X2 = pred_generator[batch_idx]
     13 
     14     # CNN features per unique patient in this batch

/tmp/ipykernel_11/3180251523.py in __getitem__(self, index)
     28 
     29         patients = self.train.loc[list_IDs_temp, "Patient"].unique()
---> 30         imgs = _read(self.img_path, patients=patients, desired_size=self.desired_size)
     31         return self.train.loc[list_IDs_temp, :].reset_index(drop=True), np.transpose(
     32             imgs, (0, 2, 3, 4, 1)

/tmp/ipykernel_11/2460121262.py in _read(path, patients, desired_size)
     16         # Read one dicom for intercept/slope
     17         dicom0 = pydicom.dcmread(os.path.join(patient_dir, files[0]))
---> 18         df = pd.DataFrame(files).apply(lambda x: os.path.join(patient_dir, x))
     19         df = convertohu(
     20             df.iloc[:, 0].apply(lambda x: pydicom.dcmread(x).pixel_array), dicom0

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/2460121262.py in <lambda>(x)
     16         # Read one dicom for intercept/slope
     17         dicom0 = pydicom.dcmread(os.path.join(patient_dir, files[0]))
---> 18         df = pd.DataFrame(files).apply(lambda x: os.path.join(patient_dir, x))
     19         df = convertohu(
     20             df.iloc[:, 0].apply(lambda x: pydicom.dcmread(x).pixel_array), dicom0

/usr/lib/python3.11/posixpath.py in join(a, *p)

/usr/lib/python3.11/genericpath.py in _check_arg_types(funcname, *args)

TypeError: join() argument must be str, bytes, or os.PathLike object, not 'Series'

## === cell 13
y_pred_list = []

for batch_idx in range(len(pred_generator)):
    X1, X2 = pred_generator[batch_idx]
    out_imgs = _get_first_output(
        CNN(tf.convert_to_tensor(X2, dtype=tf.float32))
    )  # (n_patients, latent)

    patients_unique_order = list(X1["Patient"].unique())
    feat_by_patient = {
        patients_unique_order[i]: out_imgs[i] for i in range(len(patients_unique_order))
    }

    X_imgs = tf.stack([feat_by_patient[p] for p in X1["Patient"].values], axis=0)

    X_tab = tf.convert_to_tensor(np.asarray(X1[SELECTED_COLUMNS], dtype=np.float32))
    mlp_inp = tf.concat([X_imgs, X_tab], axis=1)
    out = _get_first_output(MLP(mlp_inp))
    y_pred_list.append(out.numpy())

y_prediction = np.concatenate(y_pred_list, axis=0)

if y_prediction.shape[0] != len(X_prediction):
    raise RuntimeError(
        f"Prediction length mismatch: got {y_prediction.shape[0]} rows, expected {len(X_prediction)}"
    )



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4253616674.py in <cell line: 0>()
      3 
      4 for batch_idx in range(len(pred_generator)):
----> 5     X1, X2 = pred_generator[batch_idx]
      6     out_imgs = _get_first_output(
      7         CNN(tf.convert_to_tensor(X2, dtype=tf.float32))

/tmp/ipykernel_11/3180251523.py in __getitem__(self, index)
     28 
     29         patients = self.train.loc[list_IDs_temp, "Patient"].unique()
---> 30         imgs = _read(self.img_path, patients=patients, desired_size=self.desired_size)
     31         return self.train.loc[list_IDs_temp, :].reset_index(drop=True), np.transpose(
     32             imgs, (0, 2, 3, 4, 1)

/tmp/ipykernel_11/2460121262.py in _read(path, patients, desired_size)
     16         # Read one dicom for intercept/slope
     17         dicom0 = pydicom.dcmread(os.path.join(patient_dir, files[0]))
---> 18         df = pd.DataFrame(files).apply(lambda x: os.path.join(patient_dir, x))
     19         df = convertohu(
     20             df.iloc[:, 0].apply(lambda x: pydicom.dcmread(x).pixel_array), dicom0

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/2460121262.py in <lambda>(x)
     16         # Read one dicom for intercept/slope
     17         dicom0 = pydicom.dcmread(os.path.join(patient_dir, files[0]))
---> 18         df = pd.DataFrame(files).apply(lambda x: os.path.join(patient_dir, x))
     19         df = convertohu(
     20             df.iloc[:, 0].apply(lambda x: pydicom.dcmread(x).pixel_array), dicom0

/usr/lib/python3.11/posixpath.py in join(a, *p)

/usr/lib/python3.11/genericpath.py in _check_arg_types(funcname, *args)

TypeError: join() argument must be str, bytes, or os.PathLike object, not 'Series'

## === cell 14
sub = pd.DataFrame(
    {
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": y_prediction[:, 1].astype(np.float32),
        "Confidence": (y_prediction[:, 2] - y_prediction[:, 0]).astype(np.float32),
    }
)

sub["Confidence"] = sub["Confidence"].clip(lower=70.0)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/819489446.py in <cell line: 0>()
      4     {
      5         "Patient_Week": X_prediction["Patient_Week"].values,
----> 6         "FVC": y_prediction[:, 1].astype(np.float32),
      7         "Confidence": (y_prediction[:, 2] - y_prediction[:, 0]).astype(np.float32),
      8     }

NameError: name 'y_prediction' is not defined

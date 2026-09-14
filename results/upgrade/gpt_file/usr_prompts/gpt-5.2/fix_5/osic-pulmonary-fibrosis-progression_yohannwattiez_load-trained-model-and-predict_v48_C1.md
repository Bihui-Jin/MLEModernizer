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

-6.988567136388355

# 6. Current score

-17.28617

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -11.48628) has done: 'I fix the early import/runtime failure coming from `pydicom` by making CT-related imports lazy and fully optional, so the notebook can still run end-to-end in environments where `pydicom` (or its protobuf dependency) is broken. I also fix the missing SavedModel error by falling back to a tabular-only baseline when the `/kaggle/input/3d-cnn-mlp/*` models are not present, instead of crashing and leaving `cnn_layer` undefined. Finally, I ensure the submission is always produced with the exact required columns and a `.csv` suffix, using a reasonable default confidence (>=70) and per-patient linear trend estimated from training to improve score over a pure baseline FVC constant.'
- What this solution (achieved -17.28617) has done: 'I fix the immediate runtime failure in the first cell by avoiding the `pydicom` import path that triggers the protobuf `MessageFactory.GetPrototype` crash, and instead force the safe tabular-only fallback unless DICOM reading is explicitly enabled and `pydicom` imports cleanly. Then I keep the existing fallback logic but slightly improve it (score-moving toward your target) by calibrating the per-row `Confidence` from train residuals (still clipped to >=70), which better matches the Laplace log-likelihood metric without changing the overall modeling approach. Finally, I ensure the pipeline always reaches `submission.csv` creation with correct columns and row alignment even when the CNN/SavedModel assets are missing. All other core logic (feature prep, trend model, prediction formula) stays the same.'

# 9. Code solution

## === cell 0
import os
import random
import pathlib
import warnings

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras as K

from math import ceil

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

warnings.filterwarnings("ignore")

USE_DICOM = os.environ.get("USE_DICOM", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
)

PYDICOM_OK = False
PYDICOM_IMPORT_ERROR = None
pydicom = None
if USE_DICOM:
    try:
        import pydicom as _pydicom  # noqa: F401

        pydicom = _pydicom
        PYDICOM_OK = True
    except Exception as e:
        PYDICOM_OK = False
        PYDICOM_IMPORT_ERROR = repr(e)
else:
    PYDICOM_OK = False
    PYDICOM_IMPORT_ERROR = "USE_DICOM=0 (forced tabular-only fallback)"

print("USE_DICOM:", USE_DICOM)
print("PYDICOM_OK:", PYDICOM_OK)
if not PYDICOM_OK:
    print(
        "DICOM path disabled/unavailable; will use tabular-only fallback. Info:",
        PYDICOM_IMPORT_ERROR,
    )



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
    X_prediction.merge(raw_test, how="left", left_on="Patient", right_on="Patient")
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



## === cell 4
X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]



## === cell 5
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
        new_columns = []
        for j in range(len(categories)):
            new_columns.append("{}_{}".format(name, categories[j]))
        return new_columns


def standardisation(x, u, s):
    return (x - u) / s


def normalization(x, ma, mi):
    denom = (ma - mi) if (ma - mi) != 0 else 1.0
    return (x - mi) / denom


class data_preparation:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        try:
            self.onehotenc_smok = OneHotEncoder(
                sparse_output=True, handle_unknown="ignore"
            )
        except TypeError:
            self.onehotenc_smok = OneHotEncoder(sparse=True, handle_unknown="ignore")
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
                data["Base_percent"] = standardisation(
                    data["Base_percent"], self.base_percent_mean, self.base_percent_std
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
                self.base_week_std = data["Base_week"].std()
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )

                self.base_fvc_mean = data["Base_FVC"].mean()
                self.base_fvc_std = data["Base_FVC"].std()
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )

                self.base_percent_mean = data["Base_percent"].mean()
                self.base_percent_std = data["Base_percent"].std()
                data["Base_percent"] = standardisation(
                    data["Base_percent"], self.base_percent_mean, self.base_percent_std
                )

                self.age_mean = data["Age"].mean()
                self.age_std = data["Age"].std()
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)

                self.weeks_mean = data["Weeks"].mean()
                self.weeks_std = data["Weeks"].std()
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




## === cell 6
train_for_prep = raw_train.copy()
train_for_prep = train_for_prep.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
train_for_prep["Weeks"] = train_for_prep["Min_week"]  # keep required column present
train_for_prep["Base_week"] = 0

data_prep = data_preparation(bool_normalization=True, bool_standard=False)
_ = data_prep(
    train_for_prep[
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

X_prediction = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)



## === cell 7
if PYDICOM_OK:
    from scipy.ndimage import zoom
    import scipy.ndimage as ndimage
    from skimage import measure, morphology, segmentation

    class ConvertToHU:
        def __call__(self, imgs, dicom):
            intercept = dicom.RescaleIntercept
            slope = dicom.RescaleSlope
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
            maxg = np.max(sobel_gradient)
            if maxg != 0:
                sobel_gradient *= 255.0 / maxg

            try:
                watershed = morphology.watershed(sobel_gradient, marker_watershed)
            except Exception:
                from skimage.segmentation import watershed as seg_watershed

                watershed = seg_watershed(sobel_gradient, marker_watershed)

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
            outline += ndimage.black_tophat(outline, structure=blackhat_struct)

            lungfilter = np.bitwise_or(marker_internal, outline)
            lungfilter = ndimage.morphology.binary_closing(
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



## === cell 8
if PYDICOM_OK:

    def sort_function(x):
        return int(str(x).split(".")[0])

    def _read(path, patients=[], desired_size=(60, 512, 512)):
        X = np.empty(
            np.concatenate(([len(patients), 1], np.array(desired_size))),
            dtype=np.float32,
        )
        i = 0
        for patient in patients:
            patient_dir = os.path.join(path, patient)
            files = os.listdir(patient_dir)
            dicom = pydicom.dcmread(os.path.join(patient_dir, files[0]))

            list_patient_files = sorted(files, key=sort_function)
            df_paths = [os.path.join(patient_dir, f) for f in list_patient_files]

            df_imgs = convertohu(
                pd.Series(df_paths).apply(lambda x: pydicom.dcmread(x).pixel_array),
                dicom,
            )

            df_imgs = zoom(
                df_imgs,
                np.array(DESIRED_SIZE) / np.array(df_imgs.shape),
                mode="nearest",
            )
            X[i, 0, :, :, :] = zerocenter(
                normalize(maskwatershed(clip(df_imgs), dicom))
            )
            i += 1
        return X

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
            indices = self.indices[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            list_IDs_temp = [self.list_IDs[k] for k in indices]

            patients = self.train.loc[list_IDs_temp, "Patient"].unique()
            imgs = _read(
                self.img_path, patients=patients, desired_size=self.desired_size
            )

            X1 = self.train.loc[list_IDs_temp, :].reset_index(drop=True)
            X2 = np.transpose(imgs, (0, 2, 3, 4, 1))
            return X1, X2




## === cell 9
MODEL_DIR = "/kaggle/input/3d-cnn-mlp/model_30"
CNN_DIR = "/kaggle/input/3d-cnn-mlp/CNN_30"


def _resolve_savedmodel_dir(path):
    path = str(path)
    if os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "saved_model.pb"))
        or os.path.exists(os.path.join(path, "saved_model.pbtxt"))
    ):
        return path

    if not os.path.isdir(path):
        parent = os.path.dirname(path)
    else:
        parent = path

    for root, dirs, files in os.walk(parent):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            return root
        rel_depth = pathlib.Path(root).relative_to(parent).parts
        if len(rel_depth) >= 4:
            dirs[:] = []

    raise OSError(
        f"SavedModel file does not exist at: {path} (and could not resolve under {parent})"
    )


def _make_tfsm_layer(path):
    resolved = _resolve_savedmodel_dir(path)
    for endpoint in ["serving_default", "serve", "call"]:
        try:
            return K.layers.TFSMLayer(resolved, call_endpoint=endpoint)
        except Exception:
            pass
    loaded = tf.saved_model.load(resolved)
    sigs = list(loaded.signatures.keys())
    if not sigs:
        raise ValueError(f"No signatures found in SavedModel at {resolved}")
    return K.layers.TFSMLayer(resolved, call_endpoint=sigs[0])


MODEL_OK = True
model_layer = None
cnn_layer = None
try:
    model_layer = _make_tfsm_layer(MODEL_DIR)
    cnn_layer = _make_tfsm_layer(CNN_DIR)
    print("Resolved MODEL_DIR to:", _resolve_savedmodel_dir(MODEL_DIR))
    print("Resolved CNN_DIR to:", _resolve_savedmodel_dir(CNN_DIR))
except Exception as e:
    MODEL_OK = False
    print(
        "Pretrained SavedModels not available; will use tabular-only fallback. Error:",
        repr(e),
    )



## === cell 10
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

if PYDICOM_OK:
    pred_generator = DataGenerator(
        X_prediction,
        X_prediction.index.tolist(),
        batch_size=BATCH_SIZE,
        desired_size=DESIRED_SIZE,
        img_path=TEST_PATH,
    )




## === cell 11
def _fit_patient_slopes(train_df: pd.DataFrame):
    df = train_df[["Patient", "Weeks", "FVC"]].dropna().copy()
    slopes = {}
    for pid, g in df.groupby("Patient"):
        x = g["Weeks"].values.astype(np.float64)
        y = g["FVC"].values.astype(np.float64)
        if len(g) >= 2 and np.std(x) > 0:
            xm = x.mean()
            ym = y.mean()
            slope = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
            slopes[pid] = float(slope)
        else:
            slopes[pid] = np.nan
    slopes = pd.Series(slopes, name="slope").astype(np.float32)
    global_slope = float(np.nanmedian(slopes.values))
    slopes = slopes.fillna(global_slope)
    return slopes, global_slope


patient_slopes, global_slope = _fit_patient_slopes(raw_train)
print("Global slope (ml/week):", global_slope)


def _estimate_residual_sigma(
    train_df: pd.DataFrame, slopes: pd.Series, global_slope_val: float
) -> float:
    df = train_df[["Patient", "Weeks", "FVC"]].dropna().copy()
    base = df.sort_values(["Patient", "Weeks"]).groupby("Patient").first().reset_index()
    base = base.rename(columns={"Weeks": "BaseWeek", "FVC": "BaseFVC"})
    df = df.merge(base[["Patient", "BaseWeek", "BaseFVC"]], on="Patient", how="left")
    df["slope"] = df["Patient"].map(slopes).fillna(global_slope_val).astype(np.float32)
    pred = df["BaseFVC"].values.astype(np.float32) + df["slope"].values.astype(
        np.float32
    ) * (
        df["Weeks"].values.astype(np.float32) - df["BaseWeek"].values.astype(np.float32)
    )
    resid = (df["FVC"].values.astype(np.float32) - pred).astype(np.float32)
    mad = float(np.median(np.abs(resid)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(resid))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 250.0
    return float(sigma)


resid_sigma = _estimate_residual_sigma(raw_train, patient_slopes, global_slope)
print("Estimated residual sigma (ml):", resid_sigma)


def _build_tabular_fallback_submission(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    df["slope"] = (
        df["Patient"].map(patient_slopes).fillna(global_slope).astype(np.float32)
    )
    df["FVC_pred"] = df["Base_FVC"].astype(np.float32) + df["slope"] * (
        df["Weeks"] - df["Min_week"]
    ).astype(np.float32)

    conf = np.full((len(df),), max(70.0, resid_sigma), dtype=np.float32)

    sub = pd.DataFrame(
        {
            "Patient_Week": df["Patient_Week"].values,
            "FVC": df["FVC_pred"].values.astype(np.float32),
            "Confidence": conf,
        }
    )
    return sub




## === cell 12
if PYDICOM_OK and MODEL_OK:
    y_prediction_list = []
    for bi in range(len(pred_generator)):
        X1, X2 = pred_generator[bi]

        out_imgs = cnn_layer(tf.convert_to_tensor(X2))
        if isinstance(out_imgs, dict):
            out_imgs = list(out_imgs.values())[0]
        out_imgs = tf.convert_to_tensor(out_imgs)

        patients_in_batch = list(X1["Patient"].unique())
        counts = (
            X1["Patient"].value_counts().reindex(patients_in_batch).values.astype(int)
        )

        rep = []
        for i, cnt in enumerate(counts):
            rep.append(tf.repeat(tf.reshape(out_imgs[i], (1, -1)), cnt, axis=0))
        X_imgs = tf.concat(rep, axis=0)

        X_tab = tf.convert_to_tensor(
            np.asarray(X1[SELECTED_COLUMNS]).astype(np.float32)
        )

        pred = model_layer([X_imgs, X_tab])
        if isinstance(pred, dict):
            pred = list(pred.values())[0]
        pred = tf.convert_to_tensor(pred).numpy()

        y_prediction_list.append(pred)

    y_prediction = np.concatenate(y_prediction_list, axis=0)

    if len(y_prediction) != len(X_prediction):
        raise RuntimeError(
            f"Prediction length mismatch: got {len(y_prediction)} preds for {len(X_prediction)} rows"
        )

    fvc_pred = y_prediction[:, 1].astype(np.float32)
    conf_pred = (y_prediction[:, 2] - y_prediction[:, 0]).astype(np.float32)
    conf_pred = np.clip(conf_pred, 70.0, 1000.0)

    sub = pd.DataFrame(
        {
            "Patient_Week": X_prediction["Patient_Week"].values,
            "FVC": fvc_pred,
            "Confidence": conf_pred,
        }
    )
else:
    sub = _build_tabular_fallback_submission(X_prediction)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))
assert list(sub.columns) == ["Patient_Week", "FVC", "Confidence"]
assert str("submission.csv").endswith(".csv")

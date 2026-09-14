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

-6.859352988155193

# 6. Current score

-20.77343

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -11.48598) has done: 'I first fix the environment crash coming from protobuf/TensorFlow by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, which triggers the `MessageFactory.GetPrototype` error in Kaggle. Next, since the script expects a private Kaggle dataset (`/kaggle/input/3d-cnn-mlp`) that is not available, I add a minimal, score-reasonable fallback predictor that uses only the provided tabular data to generate valid `FVC` and `Confidence` outputs in the required format. This keeps the submission generation end-to-end and avoids changing the existing CT preprocessing/model logic when the SavedModels are present. Finally, I ensure the submission writes `submission.csv` with correct columns and row alignment to `sample_submission.csv`.'
- What this solution (achieved -10.73202) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle image. Then I keep your SavedModel path probing and inference logic unchanged, but ensure the fallback path is robust and slightly better calibrated by fitting a simple per-patient linear trend on the training history (still purely tabular, no new packages) instead of using `Base_FVC` flat for all weeks. Finally, I keep the submission assembly aligned to `sample_submission.csv` exactly and always write `submission.csv` with the required columns and no missing rows.'
- What this solution (achieved -8.78543) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf override and making TensorFlow import optional so the fallback tabular path can run even if TF fails to import. I keep your SavedModel probing/inference logic intact, but guard it behind a successful TF import so execution continues robustly when the external model dataset is absent. To nudge score toward the target, I improve the tabular fallback calibration minimally by estimating a per-patient residual uncertainty from training fit errors and using that as the predicted Confidence (clipped to the competition minimum), instead of a constant 300. Finally, I ensure a valid `submission.csv` with the exact required columns and row alignment to `sample_submission.csv` is always written.'
- What this solution (achieved -20.77343) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing protobuf’s pure-Python implementation before any TensorFlow import, and also make TensorFlow fully optional so the tabular fallback always runs even if TF still fails. I keep your SavedModel probing/inference logic intact, but ensure the script never errors out on import and always reaches submission writing. To move the score toward the target (higher is better), I make a minimal, metric-aware calibration in the fallback: instead of per-patient raw RMSE as Confidence (which can be too small/large), I compute a robust global sigma from per-patient residuals and then blend it with patient-specific sigma and a small week-distance term; this improves likelihood calibration without changing the core “per-patient linear trend” logic. Finally, I keep the submission aligned exactly to `sample_submission.csv` and write `submission.csv`.'
- What this solution (achieved -24.65934) has done: 'I fix the immediate crash by making TensorFlow truly optional: remove the protobuf environment override that triggers the `MessageFactory.GetPrototype` error, and only import TensorFlow if it succeeds (so the fallback path always runs). Then I keep your existing per-patient linear trend fallback core logic, but correct a metric-alignment bug in how `Confidence` is derived: you currently output the full interval width, which effectively doubles sigma and hurts the Laplace log-likelihood. Finally, I keep the submission creation aligned to `sample_submission.csv` and ensure `submission.csv` is always written with valid values.'
- What this solution (achieved -20.77343) has done: 'I fix the runtime crash caused by importing TensorFlow in this Kaggle image (protobuf `MessageFactory.GetPrototype`), by making the TensorFlow import truly optional and delayed so the tabular fallback always runs end-to-end. Then I correct a major metric-alignment issue in the fallback: currently you construct an interval and later recompute confidence from its width, which can miscalibrate sigma; instead we keep the same per-patient linear trend but output `FVC` and `Confidence` (sigma) directly from that logic. Finally, I ensure the submission is always aligned exactly to `sample_submission.csv` and written as `submission.csv` with the required columns and no missing rows.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import pickle
import pathlib
from math import ceil

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TF_AVAILABLE = False
tf = None
K = None
L = None

from skimage.transform import resize
from scipy.ndimage import zoom
import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation

print("Setup complete. TF_AVAILABLE =", TF_AVAILABLE)



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
    denom = ma - mi
    if denom == 0:
        return x * 0.0
    return (x - mi) / denom


class data_preparation:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
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
                data["Base_percent"] = normalization(
                    data["Base_percent"], self.base_percent_max, self.base_percent_min
                )
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
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

                self.base_percent_min = data["Base_percent"].min()
                self.base_percent_max = data["Base_percent"].max()
                data["Base_percent"] = normalization(
                    data["Base_percent"], self.base_percent_max, self.base_percent_min
                )

                self.age_min = data["Age"].min()
                self.age_max = data["Age"].max()
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)

                self.weeks_min = data["Weeks"].min()
                self.weeks_max = data["Weeks"].max()
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )

        return data




## === cell 5
data_prep = data_preparation(bool_normalization=True, bool_standard=False)

train_fit = raw_train.copy()
train_fit = train_fit.rename(columns={"Weeks": "Weeks"})
train_fit = train_fit.rename(columns={"Weeks": "Weeks"})  # no-op, kept minimal

train_fit = train_fit.assign(
    Base_week=train_fit["Weeks"],
    Base_FVC=train_fit["FVC"],
    Base_percent=train_fit["Percent"],
)
train_fit = train_fit[
    [
        "Patient",
        "Base_week",
        "Base_FVC",
        "Base_percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Weeks",
    ]
]

test_fit = raw_test.rename(
    columns={"Weeks": "Weeks", "Percent": "Percent", "FVC": "FVC"}
).assign(
    Base_week=raw_test["Weeks"],
    Base_FVC=raw_test["FVC"],
    Base_percent=raw_test["Percent"],
)
test_fit = test_fit[
    [
        "Patient",
        "Base_week",
        "Base_FVC",
        "Base_percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Weeks",
    ]
]

fit_df = pd.concat([train_fit, test_fit], axis=0, ignore_index=True)
_ = data_prep(fit_df)  # fit internal encoders and min/max on full distribution

X_prediction = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)




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
        mx = np.max(sobel_gradient)
        if mx > 0:
            sobel_gradient *= 255.0 / mx

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
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128

        return marker_internal, marker_external, marker_watershed


maskwatershed = MaskWatershed(min_hu=min(clip_bounds), iterations=2)




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
def sort_function(x):
    return int(x.split(".")[0])


def _read(path, patients=[], desired_size=(60, 512, 512)):
    import pydicom

    X = np.empty(
        np.concatenate(([len(patients), 1], np.array(desired_size))), dtype=np.float32
    )
    i = 0
    for patient in patients:
        folder = os.path.join(path, patient)
        files = sorted(
            [f for f in os.listdir(folder) if f.lower().endswith(".dcm")],
            key=sort_function,
        )

        dicom0 = pydicom.dcmread(os.path.join(folder, files[0]))

        df_paths = (
            pd.DataFrame(files).iloc[:, 0].apply(lambda f: os.path.join(folder, f))
        )
        px = df_paths.apply(lambda p: pydicom.dcmread(p).pixel_array)
        hu = convertohu(px, dicom0)

        hu = zoom(hu, np.array(DESIRED_SIZE) / np.array(hu.shape), mode="nearest")
        proc = zerocenter(normalize(maskwatershed(clip(hu), dicom0)))
        X[i, 0, :, :, :] = proc
        i += 1

    return X




## === cell 11
pred_generator = None
if not TF_AVAILABLE:
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras as K  # noqa: F401
        from tensorflow.keras import layers as L  # noqa: F401

        tf.random.set_seed(SEED)
        TF_AVAILABLE = True
        print("TF:", tf.__version__)
    except Exception as e:
        tf = None
        K = None
        L = None
        TF_AVAILABLE = False
        print("WARNING: TensorFlow import failed; will run tabular fallback only.")
        print("TF import error:", repr(e))

if TF_AVAILABLE:

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

            return (
                self.train.loc[list_IDs_temp, :].reset_index(drop=True),
                np.transpose(imgs, (0, 2, 3, 4, 1)),
                patients,
            )

    pred_generator = DataGenerator(
        X_prediction,
        X_prediction.index,
        batch_size=BATCH_SIZE,
        desired_size=DESIRED_SIZE,
        img_path=TEST_PATH,
    )




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 12
def _tfsmlayer_call(layer, inputs):
    out = layer(inputs)
    if isinstance(out, dict):
        return out[sorted(out.keys())[0]]
    return out


def _find_savedmodel_dirs(root):
    root = pathlib.Path(root)
    candidates = []
    if not root.exists():
        return candidates
    for p in root.rglob("saved_model.pb"):
        candidates.append(p.parent)
    for p in root.rglob("saved_model.pbtxt"):
        candidates.append(p.parent)
    seen = set()
    out = []
    for c in candidates:
        cs = str(c)
        if cs not in seen:
            out.append(c)
            seen.add(cs)
    return out


MODEL_ROOT = "/kaggle/input/3d-cnn-mlp"
savedmodels = _find_savedmodel_dirs(MODEL_ROOT)

CNN = None
MODEL = None
USE_SAVEDMODEL = False

if TF_AVAILABLE and len(savedmodels) > 0:
    cnn_dirs = [p for p in savedmodels if "cnn" in str(p).lower()]
    model_dirs = [p for p in savedmodels if "model" in str(p).lower()]

    cnn_path = str(cnn_dirs[0] if len(cnn_dirs) else savedmodels[0])
    model_path = str(model_dirs[0] if len(model_dirs) else savedmodels[-1])

    CNN = K.layers.TFSMLayer(cnn_path, call_endpoint="serving_default")
    MODEL = K.layers.TFSMLayer(model_path, call_endpoint="serving_default")
    USE_SAVEDMODEL = True

    print("Loaded CNN SavedModel from:", cnn_path)
    print("Loaded MODEL SavedModel from:", model_path)
else:
    if not TF_AVAILABLE:
        print("WARNING: TF unavailable; proceeding with tabular baseline fallback.")
    else:
        print(
            f"WARNING: No SavedModel found under {MODEL_ROOT}. "
            "Proceeding with a tabular baseline fallback to generate a valid submission."
        )



## === cell 13
SELECTED_COLUMNS = [
    "Base_week",
    "Base_FVC",
    "Base_percent",
    "Age",
    "Sex",
    "Weeks",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]

for col in SELECTED_COLUMNS:
    if col not in X_prediction.columns:
        X_prediction[col] = 0

X_prediction = X_prediction.sort_values(["Patient", "Weeks"]).reset_index(drop=True)



## === cell 14
y_prediction = None
fallback_fvc = None
fallback_sigma = None

if USE_SAVEDMODEL:
    y_pred_rows = []
    for X1, X2, patients in pred_generator:
        out_imgs = _tfsmlayer_call(CNN, tf.convert_to_tensor(X2))

        counts = X1["Patient"].value_counts()
        feats_repeated = []
        for i, p in enumerate(patients):
            n = int(counts.get(p, 0))
            if n == 0:
                continue
            feats_repeated.append(
                tf.repeat(tf.reshape(out_imgs[i], (1, -1)), n, axis=0)
            )
        X_imgs = (
            tf.concat(feats_repeated, axis=0)
            if len(feats_repeated)
            else tf.zeros((0, tf.shape(out_imgs)[-1]))
        )

        X_tab = tf.convert_to_tensor(
            np.asarray(X1[SELECTED_COLUMNS]).astype(np.float32)
        )

        try:
            y = _tfsmlayer_call(MODEL, [X_imgs, X_tab])
        except Exception:
            y = _tfsmlayer_call(MODEL, {"input_1": X_imgs, "input_2": X_tab})

        y_np = np.asarray(y)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        y_pred_rows.append(y_np)

    y_prediction = (
        np.concatenate(y_pred_rows, axis=0)
        if len(y_pred_rows)
        else np.zeros((0, 3), dtype=np.float32)
    )

    if y_prediction.shape[0] != X_prediction.shape[0]:
        raise ValueError(
            f"Prediction row mismatch: got {y_prediction.shape[0]} preds for {X_prediction.shape[0]} rows"
        )
else:
    tr = raw_train[["Patient", "Weeks", "FVC"]].dropna().copy()

    xg = tr["Weeks"].to_numpy(dtype=float)
    yg = tr["FVC"].to_numpy(dtype=float)
    if len(xg) >= 2 and np.std(xg) > 1e-6:
        g_slope, g_intercept = np.polyfit(xg, yg, 1)
    else:
        g_slope, g_intercept = -8.0, float(np.nanmean(yg) if len(yg) else 2000.0)

    patient_stats = {}
    all_resid = []

    for p, g in tr.groupby("Patient"):
        x = g["Weeks"].to_numpy(dtype=float)
        y = g["FVC"].to_numpy(dtype=float)
        if len(x) >= 2 and np.std(x) > 1e-6:
            s, b = np.polyfit(x, y, 1)
        else:
            s, b = g_slope, float(np.mean(y) if len(y) else g_intercept)

        resid = y - (s * x + b)
        if resid.size:
            all_resid.append(resid)

        rmse = float(np.sqrt(np.mean(resid**2)) if resid.size else np.nan)
        if not np.isfinite(rmse) or rmse <= 0:
            rmse = np.nan

        patient_stats[p] = (
            float(s),
            float(b),
            rmse,
            float(np.mean(x)) if x.size else 0.0,
            float(np.std(x)) if x.size else 0.0,
        )

    if len(all_resid):
        all_resid = np.concatenate(all_resid, axis=0)
        mad = float(np.median(np.abs(all_resid - np.median(all_resid))))
        g_sigma_rob = 1.4826 * mad
        g_sigma_rmse = float(np.sqrt(np.mean(all_resid**2)))
        g_sigma = (
            g_sigma_rob
            if (np.isfinite(g_sigma_rob) and g_sigma_rob > 1e-6)
            else g_sigma_rmse
        )
    else:
        g_sigma = 300.0

    if (not np.isfinite(g_sigma)) or g_sigma <= 0:
        g_sigma = 300.0

    pat = X_prediction["Patient"].values
    w = X_prediction["Weeks"].to_numpy(dtype=float)
    base_fvc = X_prediction["Base_FVC"].to_numpy(dtype=float)
    base_week = X_prediction["Base_week"].to_numpy(dtype=float)

    fvc_pred = np.empty(len(X_prediction), dtype=float)
    sigma_pred = np.empty(len(X_prediction), dtype=float)

    for i in range(len(X_prediction)):
        p = pat[i]
        if p in patient_stats:
            s, b, rmse, mean_week, std_week = patient_stats[p]
            fvc_pred[i] = s * w[i] + b

            if np.isfinite(rmse):
                base_sigma = 0.65 * rmse + 0.35 * g_sigma
            else:
                base_sigma = g_sigma

            dist = abs(w[i] - mean_week)
            scale = std_week if (np.isfinite(std_week) and std_week > 1e-6) else 12.0
            extra = 35.0 * (dist / scale)

            sigma_pred[i] = base_sigma + extra
        else:
            fvc_pred[i] = base_fvc[i] + g_slope * (w[i] - base_week[i])
            sigma_pred[i] = g_sigma + 35.0 * (abs(w[i] - base_week[i]) / 12.0)

    sigma_pred = np.maximum(sigma_pred, 70.0)

    fallback_fvc = fvc_pred.astype(np.float32)
    fallback_sigma = sigma_pred.astype(np.float32)



## === cell 15
if USE_SAVEDMODEL:
    fvc_pred = y_prediction[:, 1].astype(float)
    raw_width = (y_prediction[:, 2] - y_prediction[:, 0]).astype(float)
    conf_pred = np.maximum(raw_width / 2.0, 70.0)
else:
    fvc_pred = fallback_fvc.astype(float)
    conf_pred = np.maximum(fallback_sigma.astype(float), 70.0)

sub = pd.DataFrame(
    {
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": fvc_pred,
        "Confidence": conf_pred,
    }
)

sample_sub = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
sub = sample_sub[["Patient_Week"]].merge(sub, on="Patient_Week", how="left")
sub["FVC"] = sub["FVC"].fillna(sample_sub.get("FVC", 2000)).astype(float)
sub["Confidence"] = sub["Confidence"].fillna(300.0).astype(float)
sub["Confidence"] = np.maximum(sub["Confidence"].values, 70.0)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))
print("File exists:", os.path.exists("submission.csv"))
print("Any NA:", sub.isna().any().to_dict())

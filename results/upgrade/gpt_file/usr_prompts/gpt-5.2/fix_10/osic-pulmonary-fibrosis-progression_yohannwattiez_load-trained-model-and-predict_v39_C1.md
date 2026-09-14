# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-6.8633804912158345

# 6. Current score

-19.83202

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -11.48602) has done: 'I fix the immediate runtime crash coming from TensorFlow/protobuf by forcing the Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` issue in this Kaggle runtime. Next, I make the model-loading step robust: the referenced `/kaggle/input/3d-cnn-mlp/...` SavedModels are not present, so I fall back to a deterministic baseline predictor that still produces a valid `submission.csv` with correct columns and row alignment. Finally, I ensure the pipeline runs end-to-end even if `pydicom` isn’t usable, without changing the overall data flow and submission formatting logic.'
- What this solution (achieved -11.47053) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing anything that may transitively import protobuf, and by restarting the Python process only if needed (in Kaggle this is equivalent to re-running from the top). Next, I keep your existing inference flow but improve the fallback predictor (used when the external SavedModels aren’t present) from “copy baseline FVC” to a simple per-patient linear trend learned from `train.csv`, which is a minimal, legitimate change that should move the score up toward your target. Finally, I keep the submission alignment logic and ensure the output file is written as `submission.csv` with the required columns and 1908 rows.'
- What this solution (achieved -19.43401) has done: 'I fix the immediate runtime crash caused by TensorFlow/protobuf (`MessageFactory.GetPrototype`) by preventing TensorFlow from importing at all (since the external SavedModels aren’t available anyway) and making the fallback predictor run without TF. Then I keep your existing “per-patient slope learned from train.csv” core fallback logic, but make it robust by fitting slopes using each patient’s baseline week as the intercept (so predictions are anchored to the provided Base_FVC/Base_week) and by using a more realistic confidence calibrated from training residuals to improve the Laplace log-likelihood toward your target. Finally, I ensure the submission is aligned exactly to `sample_submission.csv` and always writes `submission.csv` with the required columns.'
- What this solution (achieved -18.08994) has done: 'I keep your current “anchored per-patient linear slope + global fallback” predictor (core logic) but fix two small issues that are hurting the Laplace log-likelihood: (1) your Confidence currently grows too fast with week distance, which over-penalizes via `-ln(sigma)`; we instead predict a mostly-constant sigma calibrated from train residuals (still clipped at 70). (2) we lightly shrink extreme slopes toward the global slope (a simple, stable regularization) to reduce large FVC errors (Δ) without changing the model family. These are minimal, deterministic changes and should move the score up substantially from -19.43 toward your target band around -6.86. The submission formatting/alignment remains identical and still writes `submission.csv`.'
- What this solution (achieved -14.83409) has done: 'We keep your anchored per-patient linear trend predictor (same model family and inference flow) but make two minimal, score-relevant fixes: first, estimate per-patient slopes with a tiny ridge term to prevent extreme/unstable slopes that cause large Δ (capped at 1000 but still harmful). Second, calibrate a single constant Confidence (sigma) by directly maximizing the competition’s Laplace log-likelihood on train residuals (rather than MAD), which better balances the tradeoff between the linear error term and the `-log(sigma)` term and should move the score upward toward your target. Submission alignment/formatting and the “no TensorFlow” runtime behavior remain unchanged, and the script still writes `submission.csv` with the required columns and row count.'
- What this solution (achieved -16.6634) has done: 'I fix the crash by making the baseline-preparation helpers accept both raw `train.csv` (no `Base_week` columns) and already-prepared frames (with `Base_week/Base_FVC/DeltaW`). The current KeyError happens because `_prepare_train_last3()` already adds baseline columns, then `_fit_global_slope()` tries to add them again and ends up without the expected column names; I make `_fit_global_slope()` and `_fit_patient_slopes_anchored_ridge()` operate directly on the prepared frame. With that fixed, `y_prediction` be created so the submission-writing cell can run and produce `submission.csv` with the correct columns and row alignment. Core modeling logic (anchored linear slopes + shrinkage + constant sigma calibrated to the metric) is preserved.'
- What this solution (achieved -19.83202) has done: 'Your current score (-16.66) is far below the target (-6.86), so we should improve while keeping the same core “anchored linear slope + shrinkage + constant sigma” logic. The biggest likely issue is that the slope/shrinkage/sigma are calibrated on each patient’s last-3 visits, which are not aligned to the test setting (predicting from the baseline/earliest available measurement); this can badly mis-estimate slopes and confidence. I keep the same model family but change the fitting/calibration subset to “first available baseline + last 3 visits per patient” so the learned slopes are anchored like test-time and still focus on the scoring weeks. I also apply a small, deterministic clamp to extreme slopes (winsorization based on training slope quantiles) to reduce large FVC errors without changing the approach.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # kept to preserve original imports/flow

PYDICOM_AVAILABLE = True
try:
    import pydicom  # noqa: F401
except Exception as e:
    PYDICOM_AVAILABLE = False
    PYDICOM_IMPORT_ERROR = repr(e)

from math import ceil

from scipy.ndimage import zoom
import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation

print("TensorFlow: skipped (not imported due to protobuf crash in this runtime)")
print(
    "pydicom available:", PYDICOM_AVAILABLE, globals().get("PYDICOM_IMPORT_ERROR", "")
)



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
    X_prediction.merge(raw_test, how="left", on="Patient")
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
    return (x - mi) / (ma - mi)


class data_preparation:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()
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
X_prediction = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)

required_oh = ["_Currently smokes", "_Ex-smoker", "_Never smoked"]
for c in required_oh:
    if c not in X_prediction.columns:
        X_prediction[c] = 0




## === cell 6
class ConvertToHU:
    def __call__(self, imgs, dicom):
        intercept = getattr(dicom, "RescaleIntercept", 0.0)
        slope = getattr(dicom, "RescaleSlope", 1.0)
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
def sort_function(x):  # Get the files in the right order
    return int(x.split(".")[0])


def _read(path, patients=[], desired_size=(60, 512, 512)):
    """
    Reads DICOM stacks and preprocesses them.
    If pydicom import fails in this runtime, return zeros with correct shape.
    """
    X = np.empty(
        np.concatenate(([len(patients), 1], np.array(desired_size))), dtype=np.float32
    )

    if not PYDICOM_AVAILABLE:
        X.fill(0.0)
        return X

    i = 0
    for patient in patients:
        patient_dir = os.path.join(path, patient)
        files = os.listdir(patient_dir)
        files_sorted = sorted(
            [f for f in files if f.endswith(".dcm")], key=sort_function
        )
        if len(files_sorted) == 0:
            X[i, 0, :, :, :] = 0.0
            i += 1
            continue

        dicom0 = pydicom.dcmread(os.path.join(patient_dir, files_sorted[0]))
        df_paths = (
            pd.DataFrame(files_sorted)
            .iloc[:, 0]
            .apply(lambda f: os.path.join(patient_dir, f))
        )

        pix = df_paths.apply(lambda p: pydicom.dcmread(p).pixel_array)
        hu = convertohu(pix, dicom0)

        hu = zoom(hu, np.array(DESIRED_SIZE) / np.array(hu.shape), mode="nearest")

        proc = zerocenter(normalize(maskwatershed(clip(hu), dicom0)))
        X[i, 0, :, :, :] = proc.astype(np.float32)
        i += 1

    return X




## === cell 11
pred_generator = None



## === cell 12
MODEL_DIR = "/kaggle/input/3d-cnn-mlp/model_11"
CNN_DIR = "/kaggle/input/3d-cnn-mlp/CNN_11"
tfsm_model = None
tfsm_cnn = None
print("Loaded tfsm_model:", False, "from", MODEL_DIR)
print("Loaded tfsm_cnn:", False, "from", CNN_DIR)



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


def _prepare_train_with_baseline(train_df: pd.DataFrame) -> pd.DataFrame:
    g = train_df.copy()
    g["Weeks"] = g["Weeks"].astype(float)
    g["FVC"] = g["FVC"].astype(float)
    idx = (
        g.assign(absw=g["Weeks"].abs())
        .sort_values(["Patient", "absw", "Weeks"])
        .groupby("Patient", sort=False)
        .head(1)
        .index
    )
    base = g.loc[idx, ["Patient", "Weeks", "FVC"]].rename(
        columns={"Weeks": "Base_week", "FVC": "Base_FVC"}
    )
    out = g.merge(base, on="Patient", how="left")
    out["DeltaW"] = out["Weeks"] - out["Base_week"]
    out["DeltaF"] = out["FVC"] - out["Base_FVC"]
    return out.dropna(subset=["DeltaW", "DeltaF"])


def _prepare_train_first_plus_last3(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Change is score-driven and still within the same core logic:
    - Use each patient's earliest (baseline-like) row to anchor slopes as in test-time
    - Include last-3 rows to match the scoring visits distribution
    This tends to improve generalization vs fitting only on last-3 (which can mis-anchor).
    """
    t = _prepare_train_with_baseline(train_df).sort_values(["Patient", "Weeks"])
    first_idx = t.groupby("Patient", sort=False).head(1).index
    last3_idx = t.groupby("Patient", sort=False).tail(3).index
    keep_idx = first_idx.union(last3_idx)
    return t.loc[keep_idx].copy()


def _ensure_prepared(train_df: pd.DataFrame) -> pd.DataFrame:
    needed = {"Patient", "Weeks", "FVC", "Base_week", "Base_FVC", "DeltaW", "DeltaF"}
    if needed.issubset(set(train_df.columns)):
        return train_df.copy()
    return _prepare_train_with_baseline(train_df)


def _fit_global_slope(train_df: pd.DataFrame) -> float:
    t = _ensure_prepared(train_df)
    t = t[t["DeltaW"].abs() > 0]
    if len(t) == 0:
        return 0.0
    x = t["DeltaW"].to_numpy(np.float64)
    y = t["DeltaF"].to_numpy(np.float64)
    denom = np.sum(x * x)
    if denom <= 0:
        return 0.0
    b = float(np.sum(x * y) / denom)
    return b


def _fit_patient_slopes_anchored_ridge(
    train_df: pd.DataFrame, ridge: float
) -> pd.Series:
    t = _ensure_prepared(train_df)
    slopes = {}
    ridge = float(max(0.0, ridge))
    for pid, g in t.groupby("Patient"):
        x = g["DeltaW"].to_numpy(np.float64)
        y = g["DeltaF"].to_numpy(np.float64)
        nz = np.abs(x) > 0
        x = x[nz]
        y = y[nz]
        if len(x) < 2:
            continue
        denom = float(np.sum(x * x) + ridge)
        if denom <= 0:
            continue
        slopes[pid] = float(np.sum(x * y) / denom)
    return pd.Series(slopes, name="Slope")


def _laplace_llh_per_row(delta: np.ndarray, sigma: float) -> np.ndarray:
    sigma_c = max(float(sigma), 70.0)
    delta_c = np.minimum(np.abs(delta), 1000.0)
    return -(np.sqrt(2.0) * delta_c) / sigma_c - np.log(np.sqrt(2.0) * sigma_c)


def _calibrate_sigma_constant_by_metric(resid: np.ndarray) -> float:
    resid = np.asarray(resid, dtype=np.float64)
    if resid.size == 0:
        return 150.0

    sigmas = np.arange(70.0, 401.0, 5.0, dtype=np.float64)
    scores = np.array(
        [_laplace_llh_per_row(resid, s).mean() for s in sigmas], dtype=np.float64
    )
    best_sigma = float(sigmas[int(np.argmax(scores))])

    lo = max(70.0, best_sigma - 10.0)
    hi = best_sigma + 10.0
    sigmas2 = np.arange(lo, hi + 1e-9, 1.0, dtype=np.float64)
    scores2 = np.array(
        [_laplace_llh_per_row(resid, s).mean() for s in sigmas2], dtype=np.float64
    )
    best_sigma2 = float(sigmas2[int(np.argmax(scores2))])

    return float(np.clip(best_sigma2, 70.0, 500.0))


t_fit = _prepare_train_first_plus_last3(raw_train)
global_slope = _fit_global_slope(t_fit)

dw = t_fit.loc[t_fit["DeltaW"].abs() > 0, "DeltaW"].to_numpy(np.float64)
dw2_med = float(np.median(dw * dw)) if dw.size else 0.0
RIDGE = 0.05 * dw2_med

patient_slopes = _fit_patient_slopes_anchored_ridge(t_fit, ridge=RIDGE)

if len(patient_slopes) > 10:
    lo_q, hi_q = patient_slopes.quantile([0.02, 0.98]).tolist()
    patient_slopes = patient_slopes.clip(lower=float(lo_q), upper=float(hi_q))


def _score_for_alpha(alpha: float) -> float:
    alpha = float(alpha)
    slopes_shrunk = patient_slopes * (1.0 - alpha) + global_slope * alpha
    t_tmp = t_fit.copy()
    t_tmp["Slope"] = (
        t_tmp["Patient"].map(slopes_shrunk).fillna(global_slope).astype(np.float64)
    )
    pred = t_tmp["Base_FVC"].to_numpy(np.float64) + t_tmp["Slope"].to_numpy(
        np.float64
    ) * t_tmp["DeltaW"].to_numpy(np.float64)
    resid = t_tmp["FVC"].to_numpy(np.float64) - pred
    sig = _calibrate_sigma_constant_by_metric(resid)
    return float(_laplace_llh_per_row(resid, sig).mean())


alpha_grid = np.array([0.0, 0.15, 0.30, 0.45, 0.60], dtype=np.float64)
scores = np.array([_score_for_alpha(a) for a in alpha_grid], dtype=np.float64)
SHRINK_ALPHA = float(alpha_grid[int(np.argmax(scores))])

patient_slopes_shrunk = (
    patient_slopes * (1.0 - SHRINK_ALPHA) + global_slope * SHRINK_ALPHA
)

t_cal = t_fit.copy()
t_cal["Slope"] = (
    t_cal["Patient"].map(patient_slopes_shrunk).fillna(global_slope).astype(np.float64)
)
pred_cal = t_cal["Base_FVC"].to_numpy(np.float64) + t_cal["Slope"].to_numpy(
    np.float64
) * t_cal["DeltaW"].to_numpy(np.float64)
resid_cal = t_cal["FVC"].to_numpy(np.float64) - pred_cal
base_sigma = _calibrate_sigma_constant_by_metric(resid_cal)

Xp = X_prediction.copy()
Xp["Slope"] = Xp["Patient"].map(patient_slopes_shrunk).astype(np.float32)
Xp["Slope"] = Xp["Slope"].fillna(np.float32(global_slope)).astype(np.float32)

delta_w = (
    Xp["Weeks"].astype(np.float32) - Xp["Base_week"].astype(np.float32)
).to_numpy()
base_fvc = Xp["Base_FVC"].astype(np.float32).to_numpy()
slope = Xp["Slope"].astype(np.float32).to_numpy()

fvc_pred = base_fvc + slope * delta_w
conf_pred = np.full_like(delta_w, fill_value=np.float32(base_sigma), dtype=np.float32)

y_prediction = np.stack([fvc_pred, fvc_pred, fvc_pred + conf_pred], axis=1).astype(
    np.float32
)

if y_prediction.shape[0] != len(X_prediction):
    raise ValueError(
        f"Prediction rows ({y_prediction.shape[0]}) do not match submission rows ({len(X_prediction)})."
    )

print(
    "Fallback predictor used. global_slope:",
    global_slope,
    "base_sigma:",
    base_sigma,
    "shrink_alpha:",
    SHRINK_ALPHA,
    "ridge:",
    RIDGE,
    "train_fit_rows:",
    len(t_fit),
)



## === cell 14
sub = pd.DataFrame(
    {
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": y_prediction[:, 1].astype(np.float32),
        "Confidence": (y_prediction[:, 2] - y_prediction[:, 0]).astype(np.float32),
    }
)

sub["Confidence"] = sub["Confidence"].abs()
sub["Confidence"] = sub["Confidence"].clip(lower=70.0)

sample = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
sub = sample[["Patient_Week"]].merge(sub, on="Patient_Week", how="left")
if sub[["FVC", "Confidence"]].isna().any().any():
    sub["FVC"] = sub["FVC"].fillna(sample["FVC"])
    sub["Confidence"] = sub["Confidence"].fillna(sample["Confidence"].clip(lower=70.0))

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "pydicom available:",
    PYDICOM_AVAILABLE,
    "error:" if not PYDICOM_AVAILABLE else "",
    globals().get("PYDICOM_IMPORT_ERROR", ""),
)

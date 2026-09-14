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

-8.014103497880548

# 6. Current score

-23.5347

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -12.66738) has done: 'I fix the environment/runtime blockers so the notebook runs in the current Kaggle (Python 3.11 + Keras 3) runtime: (1) avoid the `pydicom` protobuf crash by lazily importing pydicom with a safe fallback and skipping CT-based inference if unavailable, (2) remove the missing external pickle dependency by fitting `data_preparation()` directly on the available `raw_test`, and (3) address NumPy deprecations (`np.int`, `np.float`) and brittle merge logic. Because Keras 3 cannot `load_model()` legacy SavedModel folders, I load them via `keras.layers.TFSMLayer` when present, and if the model assets are not available in this environment, fall back to a stable, valid baseline prediction built from `Base_FVC` with a conservative confidence (score be reasonable and far better than producing no submission). Finally, I ensure `submission.csv` is always written with the exact required columns and row count aligned to `sample_submission.csv`.'
- What this solution (achieved -16.83821) has done: 'I fix the immediate runtime crash by preventing `pydicom` (and its protobuf dependency) from being imported at module import time, since you already have a safe CT-skip fallback and this crash currently stops the entire pipeline. I also make the pydicom/imaging dependency checks more robust so the code reliably falls back to the tabular baseline path without breaking. To nudge the score upward toward the target (your current is worse than target), I minimally improve the fallback predictions by using a simple per-patient linear trend learned from `train.csv` (still tabular-only and lightweight) instead of predicting a flat `Base_FVC` for all weeks, and I set a more reasonable constant confidence. Finally, I keep the submission format/row-count checks and ensure `submission.csv` is always written end-to-end.'
- What this solution (achieved -16.83997) has done: 'I fix the immediate runtime crash in the import cell by preventing the protobuf-related `MessageFactory.GetPrototype` failure (triggered by some imaging/protobuf stack) from aborting the whole script; instead we safely gate all imaging-related imports and keep the existing tabular fallback. Then I fix a small but impactful bug in `data_preparation` where it normalizes a non-existent `Base_percent` column (should be `Percent`), which can silently degrade features and downstream predictions. Finally, I keep the current tabular per-patient linear-trend fallback but make it more faithful to the evaluation setup by anchoring predictions to each test patient’s provided baseline measurement (learn slope from train, apply to test baseline), which should improve score toward your target without changing the overall approach.'
- What this solution (achieved -14.72193) has done: 'I fix the runtime crash in the very first import cell by avoiding importing TensorFlow/Keras at module import time, since that triggers the protobuf `MessageFactory.GetPrototype` error in your environment. Then I keep the same overall pipeline but make all TensorFlow/Keras-dependent code paths conditional, so the script can still run end-to-end with the existing tabular fallback when TF is unavailable. Finally, to improve score toward your target without changing the modeling approach, I minimally enhance the fallback by learning a single robust global slope from train (instead of using per-patient slopes that are mostly missing for test IDs) and set a slightly better-calibrated constant confidence, while preserving the same submission format and row alignment checks.'
- What this solution (achieved -16.83997) has done: 'I fix the runtime crash caused by TensorFlow importing protobuf (the `MessageFactory.GetPrototype` error) by forcing a safe tabular-only fallback and never importing TensorFlow in this environment. Then I ensure pydicom/imaging code paths are fully gated so they can’t execute when those deps are unavailable, preventing secondary failures. Finally, I minimally improve the fallback predictions (to move score toward your target) by using a robust global slope learned from train plus a simple per-patient calibration using the test baseline and a better-calibrated constant confidence, while preserving the existing overall approach and submission formatting.'
- What this solution (achieved -23.5347) has done: 'I fix the failure in the confidence-model fitting by ensuring the feature columns used for ridge regression actually exist in `train_tmp` (they currently get lost due to a merge suffix collision on `Min_week`/`Base_FVC`, leaving `feat_cols` missing). The minimal change is to build the `train_tmp` feature matrix by merging baseline per-patient features with a safe suffix and then using those columns consistently, rather than relying on a merge that silently renames them. This unblocks execution so `y_conf` is defined and a valid `submission.csv` is always written with the correct columns/row count. Core modeling logic (tabular slope + ridge calibration for sigma) remains the same; this is a correctness fix that should also improve score versus the broken pipeline (no submission).'
- What this solution (achieved -23.5347) has done: 'Your current score (-23.5347) is far below the target (-8.0141), so we should improve accuracy and (especially) confidence calibration with minimal changes to your existing tabular fallback. I keep your ridge-on-features slope model and your ridge log-sigma model, but fix a key inconsistency: the sigma model is trained on features with a `_feat` suffix while inference uses non-suffixed features, which makes confidence predictions miscalibrated and hurts the metric. Then I align the sigma model’s test-time feature matrix to the same `_feat`-suffixed schema used in training (and keep the same clipping), without changing the core approach. Finally, I add small numerical stabilizers to the ridge solves to avoid occasional singularities while preserving semantics.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

tf = None
K = None
L = None
_TF_AVAILABLE = False

pydicom = None
_PYDICOM_AVAILABLE = False

zoom = None
ndimage = None
measure = morphology = segmentation = None
_IMG_DEPS_AVAILABLE = False

try:
    from scipy.ndimage import zoom as _zoom
    import scipy.ndimage as _ndimage
    from skimage import (
        measure as _measure,
        morphology as _morphology,
        segmentation as _segmentation,
    )

    zoom = _zoom
    ndimage = _ndimage
    measure = _measure
    morphology = _morphology
    segmentation = _segmentation
    _IMG_DEPS_AVAILABLE = True
except Exception as e:
    print(
        "WARNING: imaging deps import failed; will skip CT inference. Error:", repr(e)
    )
    _IMG_DEPS_AVAILABLE = False

from math import ceil

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
raw_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
X_prediction = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 2
TEST_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression/test"

DESIRED_SIZE = (30, 256, 256)
BATCH_SIZE = 256

MASK_ITERATION = 4

clip_bounds = (-1000, 200)
pre_calculated_mean = 0.02865046213070556



## === cell 3
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")[0]
X_prediction["Weeks"] = (
    X_prediction["Patient_Week"].str.extract(r".*_(.*)")[0].astype(int)
)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

raw_test_ren = raw_test.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
X_prediction = X_prediction.merge(raw_test_ren, how="left", on="Patient")[
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
].reset_index(drop=True)



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
        return [f"{name}_{c}" for c in categories]


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
                self.base_week_std = data["Base_week"].std()
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )

                self.base_fvc_mean = data["Base_FVC"].mean()
                self.base_fvc_std = data["Base_FVC"].std()
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )

                self.base_percent_mean = data["Percent"].mean()
                self.base_percent_std = data["Percent"].std()

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
data_prep = data_preparation(bool_normalization=True, bool_standard=False)
_ = data_prep(
    raw_test.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"}).assign(
        Patient_Week=lambda d: d["Patient"] + "_" + d["Min_week"].astype(str),
        Weeks=lambda d: d["Min_week"],
        Base_week=0,
    )[
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
            "Base_week",
        ]
    ]
)

X_prediction = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)




## === cell 7
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




## === cell 8
class MaskWatershed:
    def __init__(self, min_hu, iterations):
        self.min_hu = min_hu
        self.iterations = iterations

    def __call__(self, image, dicom):
        blackhat_struct = [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ]

        blackhat_struct = ndimage.iterate_structure(blackhat_struct, self.iterations)
        stack = []
        for slice_idx in range(image.shape[0]):
            sliced = image[slice_idx]
            stack.append(
                self.seperate_lungs(
                    sliced, blackhat_struct, self.min_hu, self.iterations
                )
            )
        return np.stack(stack)

    @staticmethod
    def seperate_lungs(image, blackhat_struct, min_hu=min(clip_bounds), iterations=2):
        marker_internal, marker_external, marker_watershed = (
            MaskWatershed.generate_markers(image)
        )

        sobel_filtered_dx = ndimage.sobel(image, 1)
        sobel_filtered_dy = ndimage.sobel(image, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        maxv = np.max(sobel_gradient)
        if maxv > 0:
            sobel_gradient *= 255.0 / maxv

        watershed = segmentation.watershed(sobel_gradient, marker_watershed)

        outline = ndimage.morphological_gradient(watershed, size=(3, 3)).astype(bool)
        outline += ndimage.black_tophat(outline, structure=blackhat_struct)

        lungfilter = np.bitwise_or(marker_internal, outline)
        lungfilter = ndimage.morphology.binary_closing(
            lungfilter, structure=np.ones((5, 5)), iterations=3
        )

        segmented = np.where(
            lungfilter == 1, image, min_hu * np.ones((image.shape[0], image.shape[1]))
        )
        return segmented

    @staticmethod
    def generate_markers(image, threshold=-400):
        marker_internal_labels = measure.label(
            segmentation.clear_border(image < threshold)
        )

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

        marker_watershed = np.zeros((image.shape[0], image.shape[1]), dtype=int)
        marker_watershed += marker_internal.astype(int) * 255
        marker_watershed += marker_external.astype(int) * 128

        return marker_internal, marker_external, marker_watershed


maskwatershed = MaskWatershed(min_hu=min(clip_bounds), iterations=MASK_ITERATION)




## === cell 9
class Normalize:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, image):
        image = image.astype(float)
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
    X = np.empty(
        np.concatenate(([len(patients), 1], np.array(desired_size))), dtype=np.float32
    )
    i = 0
    for patient in patients:
        patient_dir = os.path.join(path, patient)
        files = os.listdir(patient_dir)
        dicom0 = pydicom.dcmread(os.path.join(patient_dir, files[0]))
        list_patient_files = sorted(files, key=sort_function)
        df = pd.DataFrame(list_patient_files).apply(
            lambda x: os.path.join(patient_dir, x)
        )
        df = convertohu(
            df.iloc[:, 0].apply(lambda x: pydicom.dcmread(x).pixel_array), dicom0
        )
        df = zoom(df, np.array(desired_size) / np.array(df.shape), mode="nearest")
        X[i, 0, :, :, :] = zerocenter(normalize(maskwatershed(clip(df), dicom0)))
        i += 1
    return X




## === cell 11
try:
    import pydicom as _pydicom  # noqa: F401

    pydicom = _pydicom
    _PYDICOM_AVAILABLE = True
except Exception as e:
    print(
        "WARNING: pydicom import failed; will skip CT inference and use tabular fallback. Error:",
        repr(e),
    )
    pydicom = None
    _PYDICOM_AVAILABLE = False

DataGenerator = None
pred_generator = None




## === cell 12
def _load_savedmodel_as_layer(savedmodel_dir):
    raise RuntimeError("TensorFlow unavailable in this environment.")


model = None
CNN = None



## === cell 13
n_rows = len(X_prediction)
use_full_pipeline = False  # forced off due to TF import crash

train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

train_base = (
    train.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
    .rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
)
train_base["Base_week"] = 0

pid2slope = {}
for _pid, g in train.groupby("Patient"):
    g = g.sort_values("Weeks")
    w = g["Weeks"].values.astype(np.float64)
    f = g["FVC"].values.astype(np.float64)
    if len(g) >= 2 and np.std(w) > 0:
        a, _b = np.polyfit(w, f, 1)
        if np.isfinite(a):
            pid2slope[_pid] = float(a)

train_base["Slope"] = train_base["Patient"].map(pid2slope).astype(float)
train_slope_df = train_base.dropna(subset=["Slope"]).reset_index(drop=True)

slopes = [v for v in pid2slope.values() if np.isfinite(v)]
global_slope = float(np.median(slopes)) if len(slopes) else 0.0

slope_features_cols = [
    "Patient",
    "Min_week",
    "Base_FVC",
    "Percent",
    "Age",
    "Sex",
    "SmokingStatus",
    "Weeks",
    "Patient_Week",
    "Base_week",
]
train_slope_feat_raw = train_slope_df.assign(
    Weeks=train_slope_df["Min_week"],
    Patient_Week=train_slope_df["Patient"]
    + "_"
    + train_slope_df["Min_week"].astype(str),
)[slope_features_cols]

train_slope_feat = data_prep(train_slope_feat_raw).copy()
X_test_feat = X_prediction.copy()

exclude_cols = {"Patient", "Patient_Week"}
feat_cols = [c for c in train_slope_feat.columns if c not in exclude_cols]

Xtr = (
    train_slope_feat[feat_cols]
    .apply(pd.to_numeric, errors="coerce")
    .values.astype(np.float64)
)
ytr = train_slope_df["Slope"].values.astype(np.float64)

Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
Xte = np.nan_to_num(
    X_test_feat[feat_cols]
    .apply(pd.to_numeric, errors="coerce")
    .values.astype(np.float64),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

lam = 10.0  # mild regularization to avoid overfit and keep predictions stable
XtX = Xtr.T @ Xtr
w_ridge = np.linalg.solve(
    XtX + lam * np.eye(XtX.shape[0]) + 1e-9 * np.eye(XtX.shape[0]),
    Xtr.T @ ytr,
)

pred_slope = Xte @ w_ridge
pred_slope = np.where(np.isfinite(pred_slope), pred_slope, global_slope)

base_fvc_train0 = train.loc[train["Weeks"] == 0, ["Patient", "FVC"]].rename(
    columns={"FVC": "Base_FVC_train"}
)
train_with_base = train.merge(base_fvc_train0, on="Patient", how="left")
train_with_base = train_with_base.dropna(subset=["Base_FVC_train"])

if len(train_with_base) > 10:
    resid = train_with_base["FVC"].values.astype(np.float64) - (
        train_with_base["Base_FVC_train"].values.astype(np.float64)
        + global_slope * (train_with_base["Weeks"].values.astype(np.float64) - 0.0)
    )
    global_intercept = (
        float(np.median(resid[np.isfinite(resid)]))
        if np.any(np.isfinite(resid))
        else 0.0
    )
else:
    global_intercept = 0.0

weeks_arr = X_prediction["Weeks"].values.astype(np.float64)
base_fvc_arr = X_prediction["Base_FVC"].values.astype(np.float64)
min_week_arr = X_prediction["Min_week"].values.astype(np.float64)

y_fvc = base_fvc_arr + pred_slope * (weeks_arr - min_week_arr) + global_intercept

train_tmp = train.copy()

train_tmp = train_tmp.merge(
    train_base[["Patient", "Base_FVC", "Min_week"]],
    on="Patient",
    how="left",
)

train_tmp["Slope_used"] = train_tmp["Patient"].map(pid2slope).astype(float)
train_tmp["Slope_used"] = train_tmp["Slope_used"].fillna(global_slope)
train_tmp = train_tmp.dropna(subset=["Base_FVC", "Min_week"])

train_tmp["Pred"] = (
    train_tmp["Base_FVC"].values.astype(np.float64)
    + train_tmp["Slope_used"].values.astype(np.float64)
    * (
        train_tmp["Weeks"].values.astype(np.float64)
        - train_tmp["Min_week"].values.astype(np.float64)
    )
    + global_intercept
)

train_tmp_base = (
    train.sort_values(["Patient", "Weeks"])
    .groupby("Patient", as_index=False)
    .first()[["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]]
    .rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
)
train_tmp_feat_raw = train_tmp_base.assign(
    Weeks=train_tmp_base["Min_week"],
    Patient_Week=train_tmp_base["Patient"]
    + "_"
    + train_tmp_base["Min_week"].astype(str),
    Base_week=0,
)[slope_features_cols]
train_tmp_feat = data_prep(train_tmp_feat_raw)

feat_df = train_tmp_feat[["Patient"] + feat_cols].copy()
rename_map = {c: f"{c}_feat" for c in feat_cols}
feat_df = feat_df.rename(columns=rename_map)

train_tmp = train_tmp.merge(feat_df, on="Patient", how="left")

feat_cols_feat = [f"{c}_feat" for c in feat_cols]
missing_feat = [c for c in feat_cols_feat if c not in train_tmp.columns]
if missing_feat:
    raise RuntimeError(
        f"Internal error: missing merged feature columns: {missing_feat}"
    )

Xr = (
    train_tmp[feat_cols_feat]
    .apply(pd.to_numeric, errors="coerce")
    .values.astype(np.float64)
)
Xr = np.nan_to_num(Xr, nan=0.0, posinf=0.0, neginf=0.0)

yr = np.log(
    np.clip(
        np.abs(
            train_tmp["FVC"].values.astype(np.float64)
            - train_tmp["Pred"].values.astype(np.float64)
        ),
        1.0,
        1000.0,
    )
)

lam2 = 50.0
XtX2 = Xr.T @ Xr
w_ridge2 = np.linalg.solve(
    XtX2 + lam2 * np.eye(XtX2.shape[0]) + 1e-9 * np.eye(XtX2.shape[0]),
    Xr.T @ yr,
)

Xte_feat_df = X_test_feat.copy().rename(columns=rename_map)
missing_te = [c for c in feat_cols_feat if c not in Xte_feat_df.columns]
if missing_te:
    raise RuntimeError(
        f"Internal error: missing test feature columns for confidence model: {missing_te}"
    )

Xte2 = (
    Xte_feat_df[feat_cols_feat]
    .apply(pd.to_numeric, errors="coerce")
    .values.astype(np.float64)
)
Xte2 = np.nan_to_num(Xte2, nan=0.0, posinf=0.0, neginf=0.0)

log_sigma_pred = Xte2 @ w_ridge2
log_sigma_pred = np.where(np.isfinite(log_sigma_pred), log_sigma_pred, np.log(120.0))
sigma_pred = np.exp(log_sigma_pred)

y_conf = np.clip(sigma_pred, 70.0, 1000.0).astype(np.float64)



## === cell 14
y_conf = np.clip(y_conf, 70.0, 1000.0)

sub = pd.DataFrame(
    {
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": y_fvc.astype(float),
        "Confidence": y_conf.astype(float),
    }
)

sub = sub[["Patient_Week", "FVC", "Confidence"]]
assert len(sub) == len(
    pd.read_csv(
        "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
    )
), "Submission row count mismatch."

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "TF available:",
    _TF_AVAILABLE,
    "| pydicom available:",
    _PYDICOM_AVAILABLE,
    "| img deps:",
    _IMG_DEPS_AVAILABLE,
)

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

3.9

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

-6.855675335414687

# 6. Current score

-13.28792

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.82278) has done: 'I fix the immediate runtime error caused by the removed `DataFrame.append` by replacing it with `pd.concat`, keeping the same resulting dataframe. Then I fix the missing pretrained weights issue by making the script fall back to a safe, deterministic baseline predictor (patient-wise linear trend fitted from train.csv) when the `.pth` file is not available, so the notebook always runs end-to-end. I also correct the test DICOM path (`../input/...`) to the provided dataset location (`../data/...`) to avoid file-not-found errors if image inference is used. Finally, I ensure the submission is written with the exact required columns and a `.csv` suffix.'
- What this solution (achieved -9.64556) has done: 'Your current score (-7.82278) is worse than the target (-6.8557), so we should improve performance cautiously with minimal changes while keeping the same fallback “patient-wise linear trend” core logic. The biggest easy gain is to avoid using test patient IDs that have no history in train (your current fallback defaults to a flat line), by switching to a global trend estimated from train when a patient is unseen. Next, set the prediction to exactly match the known baseline row in test (already done) and make Confidence patient-specific based on residuals from the same linear fit (clipped to the metric’s minimum 70), which usually improves the Laplace likelihood versus a constant 200. These are small, deterministic changes that keep the same approach (linear regression on Weeks→FVC) but better handle unseen patients and better calibrate uncertainty.'
- What this solution (achieved -7.56376) has done: 'I fix the fallback (no-weights) path crash by replacing the invalid `x.abs()` on a NumPy array with `np.abs(x)`, which allow `test_pred` to be created so the later cells run. Then the baseline-row override and submission creation execute as intended, producing a valid `submission.csv` with the exact required columns. These changes are minimal and keep the same core “patient-wise linear trend + calibrated sigma” logic; they only correct the runtime error and ensure end-to-end execution.'
- What this solution (achieved -7.56376) has done: 'Your current score (-7.56376) is worse than the target (-6.85568), so we should improve cautiously with minimal changes while keeping the same fallback “patient-wise linear trend + calibrated sigma” core logic. The main easy gain here is to calibrate `Confidence` globally from the training residuals (and per-patient where available), then use a small safety blend toward a global sigma to avoid overconfident patients that hurt the Laplace log-likelihood. Additionally, clipping overly-large confidence values improves the log term of the metric without changing your predicted FVCs. These are deterministic, low-risk changes that preserve the same modeling approach and submission semantics.'
- What this solution (achieved -13.82389) has done: 'We keep your exact fallback “patient-wise linear trend + calibrated sigma” approach, but make the sigma calibration match the competition’s Laplace metric more directly. Specifically, we compute residuals on *absolute FVC* (not just dFVC) and fit a patient slope/intercept (still via `np.polyfit`) so the uncertainty reflects real errors; then we use a robust global sigma estimated from those residuals and blend per-patient sigma toward it to avoid overconfident predictions that hurt the log-likelihood term. We also clip confidences a bit tighter using robust quantiles (while still respecting the required [70, 1000] metric clipping behavior) to reduce unnecessary penalty from overly-large sigma. The NN path and submission format remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -14.23141) has done: 'We keep your exact “patient-wise linear trend + calibrated sigma” fallback logic, but tune only the confidence calibration to better match the Laplace metric (the easiest lever to improve from -13.82 toward -6.86 without changing modeling). Specifically, instead of converting MAE→sigma via `sqrt(2)*MAE` (which is correct for *Laplace scale b*, not for the metric’s σ), we use `sigma ≈ MAE` and compute MAE on a fairer patient-wise leave-one-out (LOO) fit so sigma reflects generalization error rather than in-sample residuals. We also apply a small, deterministic global blend and cap sigma using a robust quantile, while keeping the required [70, 1000] clipping and the baseline-week override unchanged. The NN path, data processing, and submission schema remain the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -11.84222) has done: 'We should move the score up toward the target (current -14.231 is much worse than -6.856), and the lowest-risk lever without changing your core “patient-wise linear trend” fallback is better calibration of `Confidence` to the Laplace metric. I keep your FVC prediction logic identical, but change sigma estimation from MAE to the *optimal Laplace sigma* given errors: `sigma ≈ mean(|error|) * sqrt(2)`, computed robustly from leave-one-out residuals (per patient) and also globally, then blend per-patient toward global to avoid over/under-confidence. I also cap sigma with a robust quantile computed on the same sigma scale (not MAE scale) to reduce unnecessary log-penalty, and keep the baseline-week override and submission formatting unchanged. These are minimal, deterministic changes focused only on uncertainty calibration, which directly affects the competition metric.'
- What this solution (achieved -11.84222) has done: 'Your current score (-11.84222) is well below the target (-6.85568), so we should improve (increase) it with the smallest, safest changes while keeping the same fallback “patient-wise linear trend” core logic. The biggest lever here (without touching the model or prediction formula) is to recalibrate `Confidence` to the Laplace metric using out-of-sample residuals: compute global and per-patient leave-one-out (LOO) absolute errors, convert to the metric’s optimal sigma (`sqrt(2)*MAE`), then blend per-patient sigmas toward global to avoid overconfident penalties. We also make sure `Confidence` is never below 70 (as required by clipping) and set the baseline-week confidence to a reasonable value (70) as you already do. These changes only affect uncertainty calibration and should move the score upward toward the target without altering your FVC prediction logic.'
- What this solution (achieved -11.05328) has done: 'We keep your exact prediction core (patient-wise linear Weeks→FVC fit with a global fallback) and only adjust the `Confidence` calibration, because your current score is far below the target and the metric is very sensitive to sigma. Specifically, we estimate an “optimal” global sigma directly from out-of-sample (patient-wise LOO) absolute errors and then use that same sigma for all non-baseline rows, which is a minimal, stable change that typically improves Laplace log-likelihood when per-patient sigmas are miscalibrated. We also keep the baseline-week override (FVC = base_FVC, Confidence = 70) unchanged and ensure final clipping stays within [70, 1000]. This preserves evaluation semantics and produces the same submission format while aiming to move score upward toward the target.'
- What this solution (achieved -11.05328) has done: 'We keep your exact fallback “patient-wise linear Weeks→FVC fit with global fallback” prediction logic intact, and only adjust the uncertainty (`Confidence`) because the Laplace metric is very sensitive to sigma calibration. Instead of using a single global sigma for everyone, we compute a per-patient leave-one-out (LOO) sigma where possible, then blend it toward a robust global sigma to avoid overconfident (too-small) or underconfident (too-large) patients—this is a minimal change that typically increases the score toward your target. We also apply a mild upper cap to confidences based on a robust quantile of the derived sigmas to reduce unnecessary log-penalty, while still respecting the competition’s [70, 1000] clipping. The NN path and submission writing remain unchanged, and the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved -13.28792) has done: 'We should move the score upward toward the target (-6.86) from your current (-11.05) while keeping the same core “patient-wise linear Weeks→FVC + global fallback” predictor. The most direct minimal lever is `Confidence` calibration: your current sigma blend/cap can still be noticeably miscalibrated for the Laplace metric, so we compute an out-of-sample sigma more consistently using patient-wise leave-one-out residuals aggregated to a robust global sigma, then blend per-patient sigmas slightly less toward global to preserve patient-specific uncertainty. We also replace the hard 0.90 quantile cap with a slightly higher robust cap and add a tiny lower-floor above 70 for non-baseline rows to avoid overconfidence penalties, without changing any FVC predictions or the baseline-week override. All paths and submission formatting stay the same and the script still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib.pyplot as plt

import pydicom

import torch
import torch.nn as nn



## === cell 1
DATA_ROOT = "../data/osic-pulmonary-fibrosis-progression"
TRAIN_FOLDER = os.path.join(DATA_ROOT, "train")
TEST_FOLDER = os.path.join(DATA_ROOT, "test")


def load_scan(path):  # path == TRAIN_FOLDER/patientId
    try:
        slices = [pydicom.dcmread(path + os.sep + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        files = os.listdir(path)
        files.sort()
        slices = [pydicom.dcmread(path + os.sep + s) for s in files]
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    try:
        image[image <= -2000] = 0
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)
    except Exception:
        print("HU conversion Failed!!")
    return np.array(image, dtype=np.int16)


def plot_show_slice(slices):
    if not isinstance(slices, type(np.array([]))):
        first_patient_pixels = get_pixels_hu(slices)
    else:
        first_patient_pixels = slices

    print("Number of Total Slices in this Scan:", len(slices))
    try:
        print("Shape of the the Image is:", slices.shape[1], slices.shape[2])
    except Exception:
        print("Shape of the the Image is:BLANK")
    fig = plt.figure(figsize=(10, 10))
    for i, slc in enumerate(first_patient_pixels[:16]):
        y = fig.add_subplot(4, 4, i + 1)
        y.imshow(slc, cmap="gray")
    plt.show()


def resize_along_allaxis(
    slices, target_dimensionZ=30, target_dimensionY=100, target_dimensionX=100
):
    present_dimensionZ, present_dimensionY, present_dimensionX = (
        slices.shape[0],
        slices.shape[1],
        slices.shape[2],
    )
    if (
        target_dimensionZ == present_dimensionZ
        and target_dimensionY == present_dimensionY
        and target_dimensionX == present_dimensionX
    ):
        return slices

    zoom_factorZ = float(target_dimensionZ) / float(present_dimensionZ)
    zoom_factorY = float(target_dimensionY) / float(present_dimensionY)
    zoom_factorX = float(target_dimensionX) / float(present_dimensionX)

    resize_image = scipy.ndimage.zoom(
        slices, [zoom_factorZ, zoom_factorY, zoom_factorX], mode="nearest"
    )
    return resize_image


MIN_BOUND = -1000.0
MAX_BOUND = 400.0


def image_normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.0
    image[image < 0] = 0.0
    return image


def read_image(dir_name, patientid, Z=100, Y=200, X=200):
    path = dir_name + os.sep + patientid
    slices = load_scan(path)
    try:
        image_array = get_pixels_hu(slices)
        ctimage_resizedAll = resize_along_allaxis(
            image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
        )
        image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
        return image
    except Exception:
        print("PatientId:%s couldnt be converted" % (patientid))
        return None




## === cell 2
def csv_preprocess(data):
    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])
    FE = []
    FE.append("Healthy-FVC")

    COLS = ["Sex", "SmokingStatus"]
    for col in COLS:
        for mod in data[col].unique():
            FE.append(mod)
            data[mod] = (data[col] == mod).astype(int)

    data = data[["Patient", "Weeks", "FVC", "Age"] + FE]
    data = data.sort_values(["Patient", "Weeks"], ascending=True).reset_index(drop=True)

    FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
    rename_col = {"Weeks": "base_Weeks", "FVC": "base_FVC"}
    data = data.rename(columns=rename_col)

    npData = pd.DataFrame(
        columns=["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    )

    for pid in data["Patient"].unique():
        weeks = data.loc[data["Patient"] == pid].base_Weeks
        fvc = data.loc[data["Patient"] == pid].base_FVC
        index = data.loc[data["Patient"] == pid].index
        weeks.reset_index(inplace=True, drop=True)
        fvc.reset_index(inplace=True, drop=True)
        for k in range(len(weeks)):
            npData = pd.concat(
                [npData, data.loc[data.index == index[0]]],
                ignore_index=True,
                sort=False,
            )
            npData.iloc[-1, npData.columns.get_loc("Week")] = weeks[k]
            npData.iloc[-1, npData.columns.get_loc("actual_FVC")] = fvc[k]
    npData.reset_index(inplace=True, drop=True)
    npData = npData.fillna(0)

    return npData




## === cell 3
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ds_3d_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer):
        super(ds_3d_conv, self).__init__()
        self.depthwise = nn.Conv3d(
            nin,
            nin * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=nin,
        )
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SIGMA(nn.Module):
    def __init__(self):
        super(SIGMA, self).__init__()
        self.data_net1 = nn.Sequential(
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 118),
            nn.ReLU(),
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 502),
            nn.ReLU(),
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 118),
            nn.ReLU(),
        )
        self.data_net4 = nn.Sequential(
            nn.Linear(780, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.ReLU(),
        )

    def forward(self, data_i, image_o):
        x = torch.cat((data_i, image_o), dim=-1)
        out1 = self.data_net1(x)
        out2 = torch.cat((data_i, out1), dim=-1)
        out2 = self.data_net2(out2)
        out3 = torch.cat((data_i, out2), dim=-1)
        out3 = self.data_net3(out3)
        out4 = torch.cat((x, out1, out2, out3), dim=-1)
        out = self.data_net4(out4)
        return out


class IMAGE(nn.Module):
    def __init__(
        self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=16, dropout=True
    ):
        super(IMAGE, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            in_channel = 1 if i == 0 else channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(
                        in_channel,
                        out_channel,
                        maxpool=True,
                        kernel_size=3,
                        padding=1,
                        kernels_per_layer=1,
                    ),
                )
            else:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(
                        in_channel,
                        out_channel,
                        maxpool=False,
                        kernel_size=1,
                        padding=0,
                        kernels_per_layer=1,
                    ),
                )
        self.classifier = nn.Sequential()
        if dropout is True:
            self.classifier.add_module("dropout", nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module(
            "conv_%d" % i, nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1)
        )
        self.flat = nn.Sequential(
            Flatten(),
            nn.Linear(1728, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

    @staticmethod
    def conv_layer(
        in_channel,
        out_channel,
        maxpool=True,
        kernel_size=3,
        padding=1,
        kernels_per_layer=1,
        maxpool_stride=2,
    ):
        if maxpool is True:
            layer = nn.Sequential(
                ds_3d_conv(
                    in_channel, out_channel, kernel_size, padding, kernels_per_layer
                ),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                ds_3d_conv(
                    in_channel, out_channel, kernel_size, padding, kernels_per_layer
                ),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )
        return layer

    def forward(self, image_i):
        image_o = self.feature_extractor(image_i)
        image_o = self.classifier(image_o)
        image_o = self.flat(image_o)
        return image_o


class Combined_NET(nn.Module):
    def __init__(self):
        super(Combined_NET, self).__init__()
        self.image = IMAGE()
        self.data = SIGMA()

    def forward(self, image_i, data_i):
        image_o = self.image(image_i)
        data_o = self.data(data_i, image_o)
        return data_o




## === cell 4
C1, C2 = torch.tensor(70, dtype=torch.float32), torch.tensor(1000, dtype=torch.float32)


def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    c1_same_shape = torch.ones(sigma.size(), device=y_pred.device) * C1
    sigma_clip = torch.max(sigma, c1_same_shape)
    delta = (y_true[:, 0] - fvc_pred).abs()

    c2_same_shape = torch.ones(delta.size(), device=y_pred.device) * C2
    delta = torch.min(delta, c2_same_shape)

    sq2 = torch.tensor(2.0).sqrt()
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return (metric).mean()


def qloss(y_true, y_pred):
    qs = [0.25, 0.50, 0.75]
    q = torch.tensor(np.array([qs]), device=y_pred.device, dtype=torch.float32)
    e = y_true - y_pred
    v = torch.max(q * e, (q - 1) * e)
    return v.mean()


def quartile_loss(y_true, y_pred, _lambda=0.65):
    loss = _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)
    return loss




## === cell 5
def make_eval_data(npEval, model, device="cuda"):
    x_features = npEval[
        [
            "base_Weeks",
            "base_FVC",
            "Age",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
            "Week",
            "Healthy-FVC",
        ]
    ]
    x_features = torch.tensor(x_features.values).float()
    x_patientids_name = npEval[["Patient"]].values

    unique_patients = npEval.Patient.unique()
    loaded_images = {}
    dir_name_of_patientid = TEST_FOLDER

    for unique_patient in unique_patients:
        loaded_images[unique_patient] = read_image(
            dir_name_of_patientid, unique_patient, Z=100, Y=200, X=200
        )

    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
    model.eval()
    predictions = []
    for i, patientid in enumerate(x_patientids_name):
        x_image = loaded_images[patientid[0]]
        if x_image is None:
            x_image = np.zeros((100, 200, 200), dtype=np.uint8)

        x_image = torch.tensor(x_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x_feature = x_features[i].unsqueeze(0)

        if torch.cuda.is_available() and device == "cuda":
            x_image = x_image.cuda()
            x_feature = x_feature.cuda()

        with torch.no_grad():
            prediction = model(x_image, x_feature)
        predictions.append(prediction.to("cpu").detach().numpy()[0])

    predictions = np.array(predictions)
    npEval["FVC"] = predictions[:, 1]
    npEval["Confidence"] = predictions[:, 2] - predictions[:, 0]
    return npEval




## === cell 6
data_train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
data_test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
submission = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))



## === cell 7
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
submission = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)

merge = (
    pd.merge(data_test, submission, on=["Patient"], how="left")
    .sort_values(["Patient", "Weeks_y"])
    .reset_index(drop=True)
)
merge = merge.drop(["FVC_y"], axis=1)
merge = merge.rename(
    columns={"FVC_x": "base_FVC", "Weeks_y": "Week", "Weeks_x": "base_Weeks"}
)

del data_test

data_test = merge.loc[
    :,
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Week",
    ],
]
submission = merge.loc[:, ["Patient_Week", "Patient", "Week", "base_FVC", "Confidence"]]
submission = submission.rename(columns={"base_FVC": "FVC"})



## === cell 8
data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
FE = ["Healthy-FVC"]

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"] + FE1 + ["Week"]
)
npData = pd.concat([npData, data], ignore_index=True, sort=True)
npData = npData.fillna(0)

del data_test, data
data_test = npData[
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Age",
        "Healthy-FVC",
        "Male",
        "Female",
        "Ex-smoker",
        "Never smoked",
        "Currently smokes",
        "Week",
    ]
]
del npData



## === cell 9
WEIGHTS_PATH = (
    "../input/other686/Epoch2_Score6.862776093931154_Acc0.9275418618702558.pth"
)
use_nn = os.path.exists(WEIGHTS_PATH)

if use_nn:
    model = Combined_NET()
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    test_pred = make_eval_data(
        data_test.copy(), model, device="cuda" if torch.cuda.is_available() else "cpu"
    )
else:
    tr = data_train.copy()
    tr["Weeks"] = tr["Weeks"].astype(np.float64)
    tr["FVC"] = tr["FVC"].astype(np.float64)

    global_b, global_a = 0.0, 0.0
    if len(tr) >= 2 and np.std(tr["Weeks"].values) > 1e-9:
        global_b, global_a = np.polyfit(tr["Weeks"].values, tr["FVC"].values, 1)
        global_b, global_a = float(global_b), float(global_a)

    coef_ab = {}
    pid_sigma = {}

    loo_abs_errs_global = []
    per_pid_abs_errs = {}

    for pid, grp in tr.groupby("Patient"):
        x = grp["Weeks"].values.astype(np.float64)
        y = grp["FVC"].values.astype(np.float64)

        if len(grp) >= 2 and np.std(x) > 1e-9:
            b, a = np.polyfit(x, y, 1)
            b, a = float(b), float(a)
        else:
            b, a = global_b, global_a

        coef_ab[pid] = (a, b)

        abs_errs = []
        if len(grp) >= 3 and np.std(x) > 1e-9:
            for i in range(len(x)):
                x_loo = np.delete(x, i)
                y_loo = np.delete(y, i)
                if len(x_loo) >= 2 and np.std(x_loo) > 1e-9:
                    b2, a2 = np.polyfit(x_loo, y_loo, 1)
                    yhat_i = float(a2 + b2 * x[i])
                else:
                    yhat_i = float(a + b * x[i])
                abs_errs.append(abs(float(y[i]) - yhat_i))
        else:
            yhat = a + b * x
            resid = y - yhat
            if len(resid):
                abs_errs.extend(list(np.abs(resid)))

        if len(abs_errs) == 0:
            abs_errs = [200.0]

        per_pid_abs_errs[pid] = np.asarray(abs_errs, dtype=np.float64)
        loo_abs_errs_global.extend(list(per_pid_abs_errs[pid]))

    if len(loo_abs_errs_global) == 0:
        loo_abs_errs_global = [200.0]

    global_abs = np.asarray(loo_abs_errs_global, dtype=np.float64)
    global_mae_robust = float(np.median(global_abs))
    global_sigma = float(np.sqrt(2.0) * global_mae_robust)

    all_pid_sigmas = []
    for pid, errs in per_pid_abs_errs.items():
        s = float(np.sqrt(2.0) * float(np.mean(errs)))
        all_pid_sigmas.append(s)
    if len(all_pid_sigmas) == 0:
        all_pid_sigmas = [global_sigma]

    robust_sigma_cap = float(
        np.quantile(np.asarray(all_pid_sigmas, dtype=np.float64), 0.95)
    )

    blend_w = 0.20

    for pid, errs in per_pid_abs_errs.items():
        pid_mae = float(np.mean(errs))
        s_pid = float(np.sqrt(2.0) * pid_mae)
        s_blend = float((1.0 - blend_w) * s_pid + blend_w * global_sigma)
        pid_sigma[pid] = s_blend

    global_sigma = float(min(max(global_sigma, 70.0), 1000.0))
    robust_sigma_cap = float(min(max(robust_sigma_cap, 70.0), 1000.0))

    def predict_row(pid, week):
        a, b = coef_ab.get(pid, (global_a, global_b))
        return float(a + b * float(week))

    def predict_sigma(pid):
        s = float(pid_sigma.get(pid, global_sigma))
        if s > robust_sigma_cap:
            s = robust_sigma_cap
        if s < 75.0:
            s = 75.0
        return float(s)

    test_pred = data_test.copy()
    test_pred["FVC"] = [
        predict_row(r.Patient, r.Week) for r in test_pred.itertuples(index=False)
    ]
    test_pred["Confidence"] = [
        predict_sigma(r.Patient) for r in test_pred.itertuples(index=False)
    ]



## === cell 10
for nid in test_pred.Patient.unique():
    idx = test_pred[
        (test_pred.Patient == nid) & (test_pred.Week == test_pred.base_Weeks)
    ].index.values
    if len(idx) > 0:
        test_pred.iloc[idx[0], test_pred.columns.get_loc("FVC")] = test_pred.iloc[
            idx[0], test_pred.columns.get_loc("base_FVC")
        ]
        test_pred.iloc[idx[0], test_pred.columns.get_loc("Confidence")] = 70.0

test_pred.loc[test_pred.Confidence < 70, "Confidence"] = 70.0
test_pred.loc[test_pred.Confidence > 1000, "Confidence"] = 1000.0



## === cell 11
pred_out = test_pred[["Patient", "Week", "FVC", "Confidence"]].copy()
sub_out = submission[["Patient_Week", "Patient", "Week"]].merge(
    pred_out, on=["Patient", "Week"], how="left"
)

if sub_out["FVC"].isna().any():
    base_map = submission.set_index(["Patient", "Week"])["FVC"].to_dict()
    sub_out["FVC"] = sub_out.apply(
        lambda r: (
            base_map.get((r["Patient"], r["Week"]), np.nan)
            if pd.isna(r["FVC"])
            else r["FVC"]
        ),
        axis=1,
    )
    sub_out["Confidence"] = sub_out["Confidence"].fillna(70.0)

sub_out = sub_out[["Patient_Week", "FVC", "Confidence"]]
sub_out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())

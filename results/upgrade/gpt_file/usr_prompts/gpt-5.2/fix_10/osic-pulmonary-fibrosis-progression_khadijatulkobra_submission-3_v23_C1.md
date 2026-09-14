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

-6.860062686184474

# 6. Current score

-14.9683

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.14273) has done: 'I fix the runtime error caused by deprecated `DataFrame.append()` by replacing it with `pd.concat`, keeping the same dataframe content and column order. I also remove the missing external checkpoint dependency by running the model with randomly initialized weights if no local `.pth` is found; to keep predictions valid and reasonably calibrated for the metric, I add a safe fallback that uses a simple per-patient linear trend from the baseline FVC and clips Confidence to the required minimum. Finally, I fix broken paths (`../input/...` vs `/kaggle/input/...`) and ensure the submission is written as `submission.csv` with the exact required columns aligned to `sample_submission.csv`.'
- What this solution (achieved -10.05741) has done: 'Your current score is far below the target because the model is usually running in the “no checkpoint” fallback mode, which uses a single population slope and a fixed confidence for everyone; this is too crude for the OSIC metric. I keep the same fallback core idea (linear extrapolation from baseline) but make it patient-conditional using clinical features (Age/Sex/SmokingStatus/Percent) and a per-patient baseline correction so Week=0 exactly matches base_FVC, which should improve FVC accuracy without changing evaluation semantics. I also set Confidence from the empirical residual spread of the fitted slope model and clip it to the metric’s minimum (70), which typically improves the Laplace log-likelihood compared to a constant 250. All paths and submission formatting stay the same, and the script still produces `submission.csv` end-to-end within the time limit.'
- What this solution (achieved -10.06215) has done: 'Your gap to target is about 46% (−10.057 vs −6.860), so we should improve score, but with minimal changes and without touching the CNN/model path. The main weakness in fallback mode is that it learns a slope but doesn’t learn a patient-specific intercept correction from clinical features, so FVC predictions can be systematically shifted; adding a tiny ridge regression for an intercept term (trained on week-0 residuals) keeps the same “linear from baseline” core logic but improves calibration. Second, your Confidence is a single constant; we can keep it simple but make it week-dependent using the learned slope residual scale (larger uncertainty farther from baseline), which usually improves the Laplace log-likelihood without changing semantics. Finally, we harden feature engineering so one-hot columns are always present (even if a category is absent in test), preventing silent feature mismatch.'
- What this solution (achieved -9.29155) has done: 'Your current score (-10.062) is substantially below the target (-6.860), so we should improve (increase) it with minimal, metric-aligned changes while keeping your fallback “linear-from-baseline” logic intact. The biggest easy win is to avoid training the slope model on patients with only 2 visits (very noisy slopes), and to use a robust median slope per patient plus a weighted ridge fit so patients with more visits contribute more reliably. Second, we can compute Confidence from actual per-patient residuals around that linear fit (then aggregate into a global scale) instead of from slope RMSE alone, which tends to better match the Laplace metric. All paths, submission ordering, and the CNN branch are unchanged; this only strengthens the existing fallback regression without altering its core semantics.'
- What this solution (achieved -9.00161) has done: 'You’re currently below the target (−9.29 vs −6.86, higher is better), and the biggest low-risk gain without changing your model/training loop is to make the fallback linear extrapolation better aligned to the metric and to the train distribution. I keep your exact “baseline + predicted slope * delta_week + small intercept correction” structure, but (1) fit the slope regression with an explicit intercept term (so it can learn an overall drift and reduce bias), and (2) fit the intercept correction on week-0 residuals (true baseline vs clinical baseline) rather than median residuals across all weeks, which is more consistent with how test provides baseline. Finally, I set Confidence using a Laplace-optimal scale proxy based on the global MAE of this same baseline+slopes model (and grow it with |delta_week|), which typically improves the Laplace log-likelihood more reliably than MAD-from-multiweek-residuals while keeping the same semantics and required clipping.'
- What this solution (achieved -14.9683) has done: 'Your current score (-9.00161) is below the target (-6.86006), so we should improve it with the smallest, metric-aligned changes while keeping your fallback “baseline + slope*dweek + small intercept correction” core logic intact. The biggest low-risk issue is that your intercept correction is added to every row, which can inadvertently move the baseline week away from the known base_FVC and hurt the metric; we instead anchor Week==base_Weeks exactly by applying the correction only to non-baseline weeks. Next, we make Confidence better matched to the Laplace metric by estimating a global Laplace scale from training residuals (median absolute error) and then using that as the base Confidence (still growing with |dweek| and clipped at 70), which usually improves log-likelihood without changing prediction semantics. Finally, we ensure the test predictions are aligned to `sample_submission.csv` order by explicitly reindexing `test_pred` to `sample_sub.Patient_Week` before writing.'
- What this solution (achieved -14.9683) has done: 'We keep your fallback “baseline + slope*dweek (+ optional intercept correction)” core logic intact, but fix one key bug that is currently harming it: the Percent feature for test slope prediction is being recomputed from `Healthy-FVC` (which itself is derived from Percent), creating circular/noisy values and degrading the learned slope model. We instead use the original `Percent` from `data_test` directly in the slope feature vector (and keep `Healthy-FVC` only as an auxiliary feature as you already do). Additionally, we compute the Laplace-optimal global sigma from training residuals of the *same* feature-based slope prediction (not from per-patient median slopes), so Confidence better matches the actual error distribution without changing the semantics. These are minimal, metric-aligned changes and should increase the score toward your target without touching the CNN path or submission formatting.'
- What this solution (achieved -14.9683) has done: 'Your score is far below the target (gap ≈ -8.11), so we should legitimately increase it with very small, metric-aligned changes while keeping your fallback “baseline + predicted slope * dweek (+ intercept corr off-baseline)” core logic intact. The biggest issue is that your Confidence is much too large (often hundreds to 800), which heavily penalizes the log-likelihood even when FVC errors are moderate; we calibrate Confidence from out-of-fold residuals of the same slope model and cap it to a reasonable range closer to the metric optimum. Second, we compute residuals using the same per-row prediction recipe (including intercept correction only off-baseline) so sigma matches how you actually predict. Everything else (model, data paths, prediction structure, submission formatting) stays the same.'
- What this solution (achieved -14.9683) has done: 'Your current score (-14.9683) is far below the target (-6.8601), so we should legitimately improve it with very small, metric-aligned changes while preserving your fallback “baseline + slope*dweek (+ intercept corr off-baseline)” core logic. The biggest low-risk fix is Confidence calibration: your current Confidence is capped high and grows with |dweek|, which often over-penalizes the log-likelihood; we re-calibrate Confidence to a Laplace-optimal scale using held-out (out-of-fold) residuals from the same slope model recipe, then apply a gentler week-growth and a tighter cap. Second, we compute those residuals using the same intercept-correction logic used at inference (off-baseline only), so sigma matches actual prediction behavior. Everything else (data paths, model/CNN branch, prediction structure, submission alignment) remains unchanged and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom
import scipy.ndimage

import torch
import torch.nn as nn

from tqdm.auto import tqdm

torch.manual_seed(42)
np.random.seed(42)



## === cell 1
DATA_ROOT = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_FOLDER = os.path.join(DATA_ROOT, "train")
TEST_FOLDER = os.path.join(DATA_ROOT, "test")


def load_scan(path):
    """Load and sort DICOM slices for a patient folder."""
    try:
        slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        files = os.listdir(path)
        files.sort()
        slices = [pydicom.dcmread(os.path.join(path, s)) for s in files]
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
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
        pass
    return np.array(image, dtype=np.int16)


def resize_along_allaxis(
    slices, target_dimensionZ=30, target_dimensionY=100, target_dimensionX=100
):
    present_dimensionZ, present_dimensionY, present_dimensionX = slices.shape
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
    """Return normalized uint8 3D volume."""
    path = os.path.join(dir_name, patientid)
    slices = load_scan(path)
    image_array = get_pixels_hu(slices)
    ctimage_resized = resize_along_allaxis(
        image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
    )
    image = (image_normalize(ctimage_resized) * 255.0).astype("uint8")
    return image




## === cell 2
def csv_preprocess(data):
    data = data.copy()
    data["Healthy-FVC"] = round((data["FVC"] * 100) / data["Percent"])
    FE = ["Healthy-FVC"]

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
        weeks = data.loc[data["Patient"] == pid].base_Weeks.reset_index(drop=True)
        fvc = data.loc[data["Patient"] == pid].base_FVC.reset_index(drop=True)
        index = data.loc[data["Patient"] == pid].index
        for k in range(len(weeks)):
            npData = pd.concat(
                [npData, data.loc[data.index == index[0]]],
                ignore_index=True,
                sort=False,
            )
            npData.iloc[-1, npData.columns.get_loc("Week")] = weeks[k]
            npData.iloc[-1, npData.columns.get_loc("actual_FVC")] = fvc[k]

    npData = npData.fillna(0).reset_index(drop=True)
    return npData




## === cell 3
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ds_3d_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer):
        super().__init__()
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
        super().__init__()
        self.data_net1 = nn.Sequential(
            nn.Linear(42, 64), nn.ReLU(), nn.Linear(64, 118), nn.ReLU()
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 502), nn.ReLU()
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 118), nn.ReLU()
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
        super().__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            in_channel = 1 if i == 0 else channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    f"conv_{i}",
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
                    f"conv_{i}",
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
        if dropout:
            self.classifier.add_module("dropout", nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module(
            f"conv_{i}", nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1)
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
        if maxpool:
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
        super().__init__()
        self.image = IMAGE()
        self.data = SIGMA()

    def forward(self, image_i, data_i):
        image_o = self.image(image_i)
        data_o = self.data(data_i, image_o)
        return data_o




## === cell 4
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

    for unique_patient in unique_patients:
        loaded_images[unique_patient] = read_image(
            TEST_FOLDER, unique_patient, Z=100, Y=200, X=200
        )

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model.to("cuda")
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, patientid in enumerate(x_patientids_name):
            x_image = loaded_images[patientid[0]]
            x_image = (
                torch.tensor(x_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            x_feature = x_features[i].unsqueeze(0)

            if use_cuda:
                x_image = x_image.cuda()
                x_feature = x_feature.cuda()

            prediction = model(x_image, x_feature)
            predictions.append(prediction.to("cpu").numpy()[0])

    predictions = np.array(predictions)
    npEval["FVC"] = predictions[:, 1]
    npEval["Confidence"] = predictions[:, 2] - predictions[:, 0]
    return npEval




## === cell 5
data_train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
data_test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))



## === cell 6
submission = sample_sub.copy()
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
submission = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]].rename(
    columns={"base_FVC": "FVC"}
)



## === cell 7
data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
data["Healthy-FVC"] = data["Healthy-FVC"].replace([np.inf, -np.inf], np.nan).fillna(0)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
for c in FE1:
    data[c] = 0

data.loc[data["Sex"] == "Male", "Male"] = 1
data.loc[data["Sex"] == "Female", "Female"] = 1
data.loc[data["SmokingStatus"] == "Ex-smoker", "Ex-smoker"] = 1
data.loc[data["SmokingStatus"] == "Never smoked", "Never smoked"] = 1
data.loc[data["SmokingStatus"] == "Currently smokes", "Currently smokes"] = 1

npData = pd.DataFrame(
    columns=[
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Age",
        "Percent",
        "Healthy-FVC",
    ]
    + FE1
    + ["Week"]
)
npData = pd.concat([npData, data], ignore_index=True, sort=True)
npData = npData.fillna(0)

data_test = npData[
    [
        "Patient",
        "base_Weeks",
        "base_FVC",
        "Age",
        "Percent",
        "Healthy-FVC",
        "Male",
        "Female",
        "Ex-smoker",
        "Never smoked",
        "Currently smokes",
        "Week",
    ]
].copy()



## === cell 8
model = Combined_NET()

ckpt_path = None
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith(".pth"):
            ckpt_path = os.path.join(root, f)
            break
    if ckpt_path is not None:
        break

if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    use_fallback = False
else:
    use_fallback = True  # no checkpoint available



## === cell 9
if not use_fallback:
    test_pred = make_eval_data(data_test.copy(), model, device="cuda")
else:
    test_pred = data_test.copy()
    df = data_train.sort_values(["Patient", "Weeks"]).copy()

    patient_meta = (
        df.groupby("Patient", as_index=False)
        .first()[["Patient", "Age", "Sex", "SmokingStatus", "Percent", "FVC", "Weeks"]]
        .rename(columns={"FVC": "base_FVC_train", "Weeks": "base_Weeks_train"})
    )

    slopes_rows = []
    for pid, g in df.groupby("Patient"):
        g = g.sort_values("Weeks")
        if g["Weeks"].nunique() >= 3:
            x = g["Weeks"].values.astype(float)
            y = g["FVC"].values.astype(float)
            dx = x[None, :] - x[:, None]
            dy = y[None, :] - y[:, None]
            mask = np.abs(dx) > 1e-9
            pair_slopes = (dy[mask] / dx[mask]).astype(float)
            if pair_slopes.size:
                slope = float(np.median(pair_slopes))
                slopes_rows.append((pid, slope, int(g.shape[0])))

    slopes_df = pd.DataFrame(
        slopes_rows, columns=["Patient", "slope", "n_visits"]
    ).merge(patient_meta, on="Patient", how="left")

    if len(slopes_df) == 0:
        pop_slope = -12.0
        test_pred["FVC"] = test_pred["base_FVC"].astype(float) + pop_slope * (
            (test_pred["Week"] - test_pred["base_Weeks"]).astype(float)
        )
        test_pred["Confidence"] = 250.0
    else:
        slopes_df = slopes_df.copy()
        slopes_df["Male"] = (slopes_df["Sex"] == "Male").astype(float)
        slopes_df["Female"] = (slopes_df["Sex"] == "Female").astype(float)
        slopes_df["Ex-smoker"] = (slopes_df["SmokingStatus"] == "Ex-smoker").astype(
            float
        )
        slopes_df["Never smoked"] = (
            slopes_df["SmokingStatus"] == "Never smoked"
        ).astype(float)
        slopes_df["Currently smokes"] = (
            slopes_df["SmokingStatus"] == "Currently smokes"
        ).astype(float)

        feature_cols = [
            "Age",
            "Percent",
            "base_FVC_train",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
        ]

        X = slopes_df[feature_cols].fillna(0.0).values.astype(float)
        y = slopes_df["slope"].values.astype(float)

        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd

        wgt = np.sqrt(
            np.clip(slopes_df["n_visits"].values.astype(float) - 1.0, 1.0, 20.0)
        )
        Xw = Xs * wgt[:, None]
        yw = y * wgt

        Xw_i = np.concatenate([np.ones((Xw.shape[0], 1), dtype=float), Xw], axis=1)

        lam = 1.0  # keep mild regularization
        XtX = Xw_i.T @ Xw_i
        w = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), Xw_i.T @ yw)

        pct_test = test_pred["Percent"].astype(float).replace([np.inf, -np.inf], np.nan)
        pct_test = pct_test.fillna(float(df["Percent"].median()))

        X_test = pd.DataFrame(
            {
                "Age": test_pred["Age"].astype(float).values,
                "Percent": pct_test.astype(float).values,
                "base_FVC_train": test_pred["base_FVC"].astype(float).values,
                "Male": test_pred["Male"].astype(float).values,
                "Female": test_pred["Female"].astype(float).values,
                "Ex-smoker": test_pred["Ex-smoker"].astype(float).values,
                "Never smoked": test_pred["Never smoked"].astype(float).values,
                "Currently smokes": test_pred["Currently smokes"].astype(float).values,
            }
        )[feature_cols]

        Xte = X_test.values.astype(float)
        Xtes = (Xte - mu) / sd
        Xtes_i = np.concatenate(
            [np.ones((Xtes.shape[0], 1), dtype=float), Xtes], axis=1
        )

        slope_pred = Xtes_i @ w
        slope_pred = np.clip(slope_pred, -80.0, 30.0)

        dweek = (test_pred["Week"] - test_pred["base_Weeks"]).astype(float).values
        fvc_pred = test_pred["base_FVC"].astype(float).values + slope_pred * dweek

        base_train = (
            df.sort_values(["Patient", "Weeks"])
            .groupby("Patient", as_index=False)
            .first()[
                ["Patient", "Age", "Sex", "SmokingStatus", "Percent", "FVC", "Weeks"]
            ]
            .rename(columns={"FVC": "base_FVC", "Weeks": "base_Weeks"})
        )

        base_resid_rows = []
        for pid, g in df.groupby("Patient"):
            g = g.sort_values("Weeks")
            base_row = base_train.loc[base_train["Patient"] == pid]
            if len(base_row) != 1:
                continue
            w0 = float(base_row["base_Weeks"].values[0])
            b = float(base_row["base_FVC"].values[0])
            g0 = g.loc[g["Weeks"].astype(float) == w0]
            if len(g0) == 0:
                continue
            true0 = float(g0["FVC"].astype(float).mean())
            base_resid_rows.append((pid, true0 - b))

        resid_df = pd.DataFrame(
            base_resid_rows, columns=["Patient", "base_resid"]
        ).merge(base_train, on="Patient", how="left")

        if len(resid_df) >= 10:
            resid_df["Male"] = (resid_df["Sex"] == "Male").astype(float)
            resid_df["Female"] = (resid_df["Sex"] == "Female").astype(float)
            resid_df["Ex-smoker"] = (resid_df["SmokingStatus"] == "Ex-smoker").astype(
                float
            )
            resid_df["Never smoked"] = (
                resid_df["SmokingStatus"] == "Never smoked"
            ).astype(float)
            resid_df["Currently smokes"] = (
                resid_df["SmokingStatus"] == "Currently smokes"
            ).astype(float)

            icols = [
                "Age",
                "Percent",
                "base_FVC",
                "Male",
                "Female",
                "Ex-smoker",
                "Never smoked",
                "Currently smokes",
            ]
            Xi = resid_df[icols].fillna(0.0).values.astype(float)
            yi = resid_df["base_resid"].fillna(0.0).values.astype(float)

            mu_i = Xi.mean(axis=0)
            sd_i = Xi.std(axis=0)
            sd_i[sd_i == 0] = 1.0
            Xis = (Xi - mu_i) / sd_i

            Xis_i = np.concatenate(
                [np.ones((Xis.shape[0], 1), dtype=float), Xis], axis=1
            )

            lam_i = 10.0  # strong shrinkage (keeps change stable)
            wi = np.linalg.solve(
                Xis_i.T @ Xis_i + lam_i * np.eye(Xis_i.shape[1]),
                Xis_i.T @ yi,
            )

            Xi_test = pd.DataFrame(
                {
                    "Age": test_pred["Age"].astype(float).values,
                    "Percent": pct_test.astype(float).values,
                    "base_FVC": test_pred["base_FVC"].astype(float).values,
                    "Male": test_pred["Male"].astype(float).values,
                    "Female": test_pred["Female"].astype(float).values,
                    "Ex-smoker": test_pred["Ex-smoker"].astype(float).values,
                    "Never smoked": test_pred["Never smoked"].astype(float).values,
                    "Currently smokes": test_pred["Currently smokes"]
                    .astype(float)
                    .values,
                }
            )[icols].values.astype(float)

            Xi_tests = (Xi_test - mu_i) / sd_i
            Xi_tests_i = np.concatenate(
                [np.ones((Xi_tests.shape[0], 1), dtype=float), Xi_tests], axis=1
            )
            intercept_corr = Xi_tests_i @ wi
            intercept_corr = np.clip(intercept_corr, -300.0, 300.0)
        else:
            intercept_corr = np.zeros(len(test_pred), dtype=float)

        is_baseline = (
            test_pred["Week"].astype(float).values
            - test_pred["base_Weeks"].astype(float).values
        ) == 0.0
        intercept_corr = np.asarray(intercept_corr, dtype=float)
        intercept_corr = np.where(is_baseline, 0.0, intercept_corr)

        test_pred["FVC"] = fvc_pred + intercept_corr

        rng = np.random.RandomState(42)
        unique_pids = slopes_df["Patient"].dropna().unique()
        rng.shuffle(unique_pids)
        kfold = min(5, max(2, unique_pids.shape[0] // 30))  # stable small CV
        folds = np.array_split(unique_pids, kfold)

        resid_abs_oof = []
        for k in range(kfold):
            val_pids = set(folds[k].tolist())
            trn = slopes_df.loc[~slopes_df["Patient"].isin(val_pids)].copy()
            if trn.shape[0] < 20:
                continue

            Xk = trn[feature_cols].fillna(0.0).values.astype(float)
            yk = trn["slope"].values.astype(float)

            muk = Xk.mean(axis=0)
            sdk = Xk.std(axis=0)
            sdk[sdk == 0] = 1.0
            Xks = (Xk - muk) / sdk

            wgtk = np.sqrt(
                np.clip(trn["n_visits"].values.astype(float) - 1.0, 1.0, 20.0)
            )
            Xkw = Xks * wgtk[:, None]
            ykw = yk * wgtk
            Xkw_i = np.concatenate(
                [np.ones((Xkw.shape[0], 1), dtype=float), Xkw], axis=1
            )

            wk = np.linalg.solve(
                (Xkw_i.T @ Xkw_i) + lam * np.eye(Xkw_i.shape[1]),
                Xkw_i.T @ ykw,
            )

            resid_trn = resid_df.loc[~resid_df["Patient"].isin(val_pids)].copy()
            if resid_trn.shape[0] >= 20:
                resid_trn["Male"] = (resid_trn["Sex"] == "Male").astype(float)
                resid_trn["Female"] = (resid_trn["Sex"] == "Female").astype(float)
                resid_trn["Ex-smoker"] = (
                    resid_trn["SmokingStatus"] == "Ex-smoker"
                ).astype(float)
                resid_trn["Never smoked"] = (
                    resid_trn["SmokingStatus"] == "Never smoked"
                ).astype(float)
                resid_trn["Currently smokes"] = (
                    resid_trn["SmokingStatus"] == "Currently smokes"
                ).astype(float)

                Xik = resid_trn[icols].fillna(0.0).values.astype(float)
                yik = resid_trn["base_resid"].fillna(0.0).values.astype(float)
                muik = Xik.mean(axis=0)
                sdik = Xik.std(axis=0)
                sdik[sdik == 0] = 1.0
                Xiks = (Xik - muik) / sdik
                Xiks_i = np.concatenate(
                    [np.ones((Xiks.shape[0], 1), dtype=float), Xiks], axis=1
                )
                wik = np.linalg.solve(
                    (Xiks_i.T @ Xiks_i) + lam_i * np.eye(Xiks_i.shape[1]),
                    Xiks_i.T @ yik,
                )
            else:
                muik = None
                sdik = None
                wik = None

            for pid in val_pids:
                g = df.loc[df["Patient"] == pid].sort_values("Weeks")
                if g.shape[0] < 3:
                    continue

                base_row = base_train.loc[base_train["Patient"] == pid]
                if len(base_row) != 1:
                    continue
                b = float(base_row["base_FVC"].values[0])
                w0 = float(base_row["base_Weeks"].values[0])

                meta_row = slopes_df.loc[slopes_df["Patient"] == pid]
                if len(meta_row) != 1:
                    continue
                mr = meta_row.iloc[0]

                xrow = np.array(
                    [
                        float(mr["Age"]) if pd.notna(mr["Age"]) else 0.0,
                        float(mr["Percent"]) if pd.notna(mr["Percent"]) else 0.0,
                        (
                            float(mr["base_FVC_train"])
                            if pd.notna(mr["base_FVC_train"])
                            else 0.0
                        ),
                        float(mr["Male"]),
                        float(mr["Female"]),
                        float(mr["Ex-smoker"]),
                        float(mr["Never smoked"]),
                        float(mr["Currently smokes"]),
                    ],
                    dtype=float,
                )
                xrow_s = (xrow - muk) / sdk
                s_hat = float(np.concatenate([[1.0], xrow_s], axis=0) @ wk)
                s_hat = float(np.clip(s_hat, -80.0, 30.0))

                dw = g["Weeks"].astype(float).values - w0
                pred = b + s_hat * dw

                if wik is not None:
                    br = base_row.iloc[0]
                    xirow = np.array(
                        [
                            float(br["Age"]) if pd.notna(br["Age"]) else 0.0,
                            float(br["Percent"]) if pd.notna(br["Percent"]) else 0.0,
                            float(br["base_FVC"]) if pd.notna(br["base_FVC"]) else 0.0,
                            1.0 if br["Sex"] == "Male" else 0.0,
                            1.0 if br["Sex"] == "Female" else 0.0,
                            1.0 if br["SmokingStatus"] == "Ex-smoker" else 0.0,
                            1.0 if br["SmokingStatus"] == "Never smoked" else 0.0,
                            1.0 if br["SmokingStatus"] == "Currently smokes" else 0.0,
                        ],
                        dtype=float,
                    )
                    xirow_s = (xirow - muik) / sdik
                    ic = float(np.concatenate([[1.0], xirow_s], axis=0) @ wik)
                    ic = float(np.clip(ic, -300.0, 300.0))
                    pred = pred + np.where(np.abs(dw) < 1e-9, 0.0, ic)

                resid_abs_oof.extend(
                    np.abs(g["FVC"].astype(float).values - pred).tolist()
                )

        if len(resid_abs_oof) >= 200:
            med_ae = float(np.median(resid_abs_oof))
            base_sigma = float(np.sqrt(2.0) * (med_ae / np.log(2.0)))
            base_sigma = float(np.clip(base_sigma, 70.0, 180.0))
        else:
            resid_abs = []
            for i, row in slopes_df.iterrows():
                pid = row["Patient"]
                g = df.loc[df["Patient"] == pid].sort_values("Weeks")
                if g.shape[0] < 3:
                    continue

                xrow = np.array(
                    [
                        float(row["Age"]) if pd.notna(row["Age"]) else 0.0,
                        float(row["Percent"]) if pd.notna(row["Percent"]) else 0.0,
                        (
                            float(row["base_FVC_train"])
                            if pd.notna(row["base_FVC_train"])
                            else 0.0
                        ),
                        float(row["Male"]),
                        float(row["Female"]),
                        float(row["Ex-smoker"]),
                        float(row["Never smoked"]),
                        float(row["Currently smokes"]),
                    ],
                    dtype=float,
                )
                xrow_s = (xrow - mu) / sd
                s_hat = float(np.concatenate([[1.0], xrow_s], axis=0) @ w)
                s_hat = float(np.clip(s_hat, -80.0, 30.0))

                base_row = base_train.loc[base_train["Patient"] == pid]
                if len(base_row) != 1:
                    continue
                b = float(base_row["base_FVC"].values[0])
                w0 = float(base_row["base_Weeks"].values[0])

                dw = g["Weeks"].astype(float).values - w0
                pred = b + s_hat * dw
                resid_abs.extend(np.abs(g["FVC"].astype(float).values - pred).tolist())

            if len(resid_abs) >= 50:
                med_ae = float(np.median(resid_abs))
                base_sigma = float(np.sqrt(2.0) * (med_ae / np.log(2.0)))
                base_sigma = float(np.clip(base_sigma, 70.0, 200.0))
            else:
                base_sigma = 140.0

        step = float(np.clip(base_sigma / 60.0, 0.5, 6.0))
        conf = base_sigma + np.abs(dweek) * step
        test_pred["Confidence"] = np.clip(conf, 70.0, 200.0)



## === cell 10
for nid in test_pred.Patient.unique():
    idx = test_pred[
        (test_pred.Patient == nid) & (test_pred.Week == test_pred.base_Weeks)
    ].index.values
    if len(idx) > 0:
        test_pred.iloc[idx[0], test_pred.columns.get_loc("FVC")] = float(
            test_pred.iloc[idx[0], test_pred.columns.get_loc("base_FVC")]
        )
        test_pred.iloc[idx[0], test_pred.columns.get_loc("Confidence")] = 70.0

test_pred.loc[test_pred.Confidence < 70, "Confidence"] = 70.0



## === cell 11
pred_map = test_pred.copy()
pred_map["Patient_Week"] = (
    pred_map["Patient"].astype(str) + "_" + pred_map["Week"].astype(int).astype(str)
)
pred_map = pred_map.set_index("Patient_Week")[["FVC", "Confidence"]]

out = sample_sub.copy()
out = out.join(pred_map, on="Patient_Week", rsuffix="_pred")

out["FVC"] = out["FVC"].fillna(2000.0).astype(float)
out["Confidence"] = out["Confidence"].fillna(250.0).astype(float)
out["Confidence"] = out["Confidence"].clip(lower=70.0)

out.to_csv("submission.csv", index=False)
print(out.head())
print("Wrote submission.csv with shape:", out.shape)

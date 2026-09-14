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

-6.909822822950186

# 6. Current score

-7.91585

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.82278) has done: 'I fix the immediate runtime error by replacing the removed `DataFrame.append` with `pd.concat` while keeping the same dataframe construction. Then I address the missing model weights file by falling back to a simple, deterministic baseline (linear extrapolation from the provided baseline FVC/Percent) so the notebook always runs end-to-end in this Kaggle environment. I also make the DICOM dependency optional (only used if the external weights are present) to avoid failures on environments without `pydicom`. Finally, I ensure the submission is written as `submission.csv` with the exact required columns and row alignment to `sample_submission.csv`.'
- What this solution (achieved -8.93093) has done: 'Your current fallback (when pretrained weights aren’t available) predicts a constant FVC equal to baseline for all weeks, which is likely leaving score on the table versus your target. I keep the same overall pipeline and submission alignment, but improve only the fallback by fitting a simple per-patient linear trend (FVC vs Weeks) from the provided training history and using that slope plus the test baseline point to extrapolate future weeks. I also set the fallback Confidence using the training residual spread (clipped to Kaggle’s minimum 70), which better matches the metric than a fixed 200 while remaining deterministic and lightweight. These changes are minimal, do not alter the pretrained-path behavior, and should move your score upward toward the target.'
- What this solution (achieved -8.43057) has done: 'I keep your pretrained-weights path unchanged and only adjust the fallback (no-weights) branch, since that’s what is driving your current -8.93 score. The minimal improvement is to replace the “use a generic patient slope” extrapolation with a stronger but still-simple clinical baseline: predict each patient’s FVC using a global linear model trained on `train.csv` with features you already use (Age/Sex/Smoking/Percent/Weeks + patient baseline), then anchor predictions exactly at the provided test baseline week. I also compute Confidence from the training residual distribution of that same model (then clip to ≥70), which is aligned with the Laplace-like metric and typically improves score without changing the overall pipeline. These changes preserve your core semantics (deterministic, lightweight, no extra packages, same output format) and should move the score upward toward the target band.'
- What this solution (achieved -8.00076) has done: 'Your current fallback uses a global linear regression but doesn’t include the `Healthy-FVC` feature you already compute and that’s central to your pipeline; adding it as an extra regressor is a minimal change that should improve FVC fit and move the score upward toward your target. I also compute Confidence using the competition’s Laplace form (estimate the best constant sigma from training residuals via `mean(|resid|)*sqrt(2)`, then clip to ≥70), which is directly aligned with the evaluation metric and typically yields a better score than using the median absolute residual. Finally, I keep the pretrained-weights path unchanged and preserve submission alignment exactly to `sample_submission.csv`.'
- What this solution (achieved -8.24174) has done: 'We’re currently below the target (score -8.00076 vs target -6.9098; higher is better), so we want a small, low-risk boost without changing your core model path. The biggest gain with minimal change is to calibrate the fallback branch’s Confidence better for the Laplace metric: a single global sigma is often suboptimal, so we compute a per-week (Weeks offset) sigma from train residuals, then map those sigmas onto the submission weeks and clip to ≥70. This keeps the same linear-regression fallback and the same predictions anchoring at baseline, but typically improves the metric by reducing the log-penalty from over/under-confidence. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved -7.98167) has done: 'We’re currently below the target (current -8.24174 vs target -6.9098; higher is better), so we want a small, low-risk improvement without changing your pretrained branch or overall pipeline. The biggest likely gain with minimal logic change is to make the linear fallback fit more robust by adding ridge-style regularization to the closed-form least-squares (same linear model/feature set, just stabilized coefficients), which often reduces extrapolation error on unseen patients. In addition, we compute the per-week sigma more robustly by shrinking sparse week-specific sigmas toward the global sigma (reducing overconfident/underconfident weeks), which is directly aligned with the Laplace-like metric. Everything else (feature engineering, anchoring to baseline, submission alignment/format) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved -7.88824) has done: 'We’re currently below the target (current -7.98167 vs target -6.90982; higher is better), so we want a small, low-risk boost without changing your model architecture or overall pipeline. The biggest minimal lever left in your fallback branch is aligning the confidence calibration with the Laplace metric: instead of using only week-specific residuals, we compute the optimal constant sigma per predicted week-gap (Week − base_Weeks) and shrink it toward a global sigma to avoid noisy bins, then map that to test rows. This keeps the same ridge-stabilized linear regressor and the same baseline anchoring, but typically improves the score by reducing the log-likelihood penalty from miscalibrated confidence. Everything remains deterministic, runs end-to-end, and still writes a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved -7.92929) has done: 'We’re below the target (current -7.888 vs target -6.910; higher is better), so we want a small, low-risk boost without touching the pretrained CNN path. The weakest part left is the fallback confidence calibration: using only gap-specific residuals ignores that residual scale varies strongly by `Weeks` and can be sparse/noisy per gap. I keep the exact same ridge linear regressor and baseline anchoring, but change Confidence to a blended estimate that combines (a) gap-based sigma and (b) absolute-week-based sigma, both shrunk toward the same global sigma. This typically improves the Laplace log-likelihood by better matching uncertainty without changing FVC predictions (only Confidence), moving the score upward toward your target.'
- What this solution (achieved -7.91586) has done: 'We’re currently below the target (current -7.92929 vs target -6.90982; higher is better), so we want a small, low-risk score increase while keeping your model/fallback logic intact. The biggest minimal lever left is Confidence calibration: your current confidence blend uses a fixed 50/50 weight and then forces baseline rows to 70, which can be suboptimal under the Laplace log-likelihood. I keep your ridge-regression FVC predictions unchanged and only tune the Confidence blending weight using an internal score proxy on the training residuals (same metric form, using the same per-gap and per-week sigma tables), then apply that single learned weight to test. This is deterministic, fast, and should move the score upward toward the target without changing architecture, training, or feature extraction.'
- What this solution (achieved -7.91585) has done: 'We’re below the target (current -7.91586 vs target -6.90982; higher is better), so we want a small, low-risk improvement without changing your FVC prediction model. The most leverage with minimal change is to calibrate Confidence more like the evaluation metric’s optimum per row: for Laplace log-likelihood, the best sigma is approximately `sqrt(2)*|residual|`, so we can learn a simple multiplicative calibration `c` on your already-computed blended sigma (gap/week blend) using training residuals and the exact metric formula. This keeps the ridge regressor, feature set, and your learned blend weight `best_w` intact; it only rescales Confidence deterministically and then clips to ≥70 as required. Finally, we keep the baseline-week override to preserve alignment with the provided baseline measurement.'
- What this solution (achieved -7.91585) has done: 'Your current score (-7.91585) is below the target (-6.90982), so we want a small, low-risk improvement without changing your FVC prediction model. The most direct lever left is Confidence calibration: right now it’s capped at 400 and also forcibly set to 70 at the baseline week, which can make you overconfident and get heavily penalized when FVC errors are moderate/large. I keep the ridge regression and all feature logic identical, but (1) remove the hard upper cap of 400 (Kaggle only clips the *minimum* at 70), and (2) learn a slightly better global rescale `best_c` over a wider range while matching the exact competition metric. These are minimal changes focused only on the metric-aligned uncertainty output and should move the score upward toward your target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    import pydicom  # noqa: F401
    import scipy.ndimage  # noqa: F401

    _HAS_DICOM = True
except Exception:
    _HAS_DICOM = False

import torch
import torch.nn as nn



## === cell 1
TRAIN_FOLDER = "../data/train"


def load_scan(
    path,
):  # Here path == (../input/osic-pulmonary-fibrosis-progression/train/patientId)
    slices = [pydicom.dcmread(path + os.sep + s) for s in os.listdir(path)]
    try:
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
    image_array = get_pixels_hu(slices)
    ctimage_resizedAll = resize_along_allaxis(
        image_array, target_dimensionX=X, target_dimensionY=Y, target_dimensionZ=Z
    )
    image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
    return image




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
            nn.Linear(64, 2),
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
    dir_name_of_patientid = "../input/osic-pulmonary-fibrosis-progression/test"

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
    npEval["Confidence"] = predictions[:, 0]
    return npEval




## === cell 5
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 6
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

submission = merge.loc[:, ["Patient_Week"]].copy()
submission["FVC"] = np.nan
submission["Confidence"] = np.nan



## === cell 7
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
].copy()
del npData



## === cell 8
weights_path = (
    "../input/ww-7-679/Epoch7_Score6.791304574107492_Acc0.9315211807655183.pth"
)

use_pretrained = _HAS_DICOM and os.path.exists(weights_path)

if use_pretrained:
    model = Combined_NET()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    test_pred = make_eval_data(
        data_test.copy(), model, device="cuda" if torch.cuda.is_available() else "cpu"
    )
else:
    tr = data_train.copy()

    tr["Healthy-FVC"] = np.round(
        (tr["FVC"].astype(float) * 100.0) / tr["Percent"].astype(float)
    )

    tr_base = tr.loc[
        tr.groupby("Patient")["Weeks"].idxmin(),
        ["Patient", "Weeks", "FVC", "Healthy-FVC"],
    ]
    tr_base = tr_base.rename(
        columns={
            "Weeks": "base_Weeks",
            "FVC": "base_FVC",
            "Healthy-FVC": "base_Healthy-FVC",
        }
    ).reset_index(drop=True)
    tr = tr.merge(tr_base, on="Patient", how="left", validate="many_to_one")

    tr["Male"] = (tr["Sex"] == "Male").astype(float)
    tr["Female"] = (tr["Sex"] == "Female").astype(float)
    tr["Ex-smoker"] = (tr["SmokingStatus"] == "Ex-smoker").astype(float)
    tr["Never smoked"] = (tr["SmokingStatus"] == "Never smoked").astype(float)
    tr["Currently smokes"] = (tr["SmokingStatus"] == "Currently smokes").astype(float)

    feat_cols = [
        "base_Weeks",
        "base_FVC",
        "Age",
        "Percent",
        "base_Healthy-FVC",
        "Male",
        "Female",
        "Ex-smoker",
        "Never smoked",
        "Currently smokes",
        "Weeks",
    ]

    X = tr[feat_cols].astype(float).values
    y = tr["FVC"].astype(float).values

    X_design = np.concatenate([np.ones((X.shape[0], 1), dtype=float), X], axis=1)

    lam = 1e-3
    A = X_design.T @ X_design
    A_reg = A + lam * np.eye(A.shape[0], dtype=float)
    beta = np.linalg.solve(A_reg, X_design.T @ y)

    yhat = X_design @ beta
    resid = np.abs(y - yhat)

    tr_res = tr[["Weeks", "base_Weeks"]].copy()
    tr_res["gap"] = (
        tr_res["Weeks"].astype(int) - tr_res["base_Weeks"].astype(int)
    ).astype(int)
    tr_res["abs_resid"] = resid.astype(float)

    def laplace_metric_from_abs_resid(abs_res, sigma):
        sigma_c = np.maximum(sigma, 70.0)
        delta = np.minimum(abs_res, 1000.0)
        return -np.sqrt(2.0) * delta / sigma_c - np.log(np.sqrt(2.0) * sigma_c)

    global_mae = float(np.nanmean(resid))
    global_sigma = float(
        np.clip(
            global_mae * np.sqrt(2.0) if np.isfinite(global_mae) else 200.0,
            70.0,
            np.inf,
        )
    )

    per_gap_mae = tr_res.groupby("gap")["abs_resid"].mean()
    per_gap_sigma_raw = (per_gap_mae * np.sqrt(2.0)).clip(lower=70.0)
    per_gap_count = tr_res.groupby("gap")["abs_resid"].size().astype(float)
    k_gap = 25.0
    per_gap_sigma = (per_gap_sigma_raw * per_gap_count + global_sigma * k_gap) / (
        per_gap_count + k_gap
    )
    per_gap_sigma = per_gap_sigma.clip(lower=70.0)

    per_week_mae = tr_res.groupby("Weeks")["abs_resid"].mean()
    per_week_sigma_raw = (per_week_mae * np.sqrt(2.0)).clip(lower=70.0)
    per_week_count = tr_res.groupby("Weeks")["abs_resid"].size().astype(float)
    k_week = 25.0
    per_week_sigma = (per_week_sigma_raw * per_week_count + global_sigma * k_week) / (
        per_week_count + k_week
    )
    per_week_sigma = per_week_sigma.clip(lower=70.0)

    sig_gap_tr = tr_res["gap"].map(per_gap_sigma).astype(float).values
    sig_week_tr = tr_res["Weeks"].astype(int).map(per_week_sigma).astype(float).values
    sig_gap_tr = np.where(np.isfinite(sig_gap_tr), sig_gap_tr, global_sigma)
    sig_week_tr = np.where(np.isfinite(sig_week_tr), sig_week_tr, global_sigma)

    w_grid = np.linspace(0.0, 1.0, 21)  # deterministic small grid search
    best_w = 0.5
    best_score = -np.inf
    abs_res_tr = tr_res["abs_resid"].astype(float).values
    for w_try in w_grid:
        sig_try = w_try * sig_gap_tr + (1.0 - w_try) * sig_week_tr
        score_try = float(np.mean(laplace_metric_from_abs_resid(abs_res_tr, sig_try)))
        if score_try > best_score:
            best_score = score_try
            best_w = float(w_try)

    sig_blend_tr = best_w * sig_gap_tr + (1.0 - best_w) * sig_week_tr

    c_grid = np.linspace(0.5, 2.0, 61)
    best_c = 1.0
    best_c_score = -np.inf
    for c_try in c_grid:
        sig_try = np.clip(c_try * sig_blend_tr, 70.0, np.inf)
        score_try = float(np.mean(laplace_metric_from_abs_resid(abs_res_tr, sig_try)))
        if score_try > best_c_score:
            best_c_score = score_try
            best_c = float(c_try)

    test0 = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")[
        ["Patient", "Percent"]
    ]
    test_feat = data_test.copy().merge(
        test0, on="Patient", how="left", validate="many_to_one"
    )
    test_feat["Percent"] = (
        test_feat["Percent"].astype(float).fillna(test_feat["Percent"].median())
    )

    test_feat = test_feat.rename(columns={"Healthy-FVC": "base_Healthy-FVC"})

    Xt = (
        test_feat[
            [
                "base_Weeks",
                "base_FVC",
                "Age",
                "Percent",
                "base_Healthy-FVC",
                "Male",
                "Female",
                "Ex-smoker",
                "Never smoked",
                "Currently smokes",
                "Week",
            ]
        ]
        .astype(float)
        .values
    )
    Xt_design = np.concatenate([np.ones((Xt.shape[0], 1), dtype=float), Xt], axis=1)

    test_pred = test_feat.copy()
    test_pred["FVC"] = Xt_design @ beta

    test_pred["gap"] = (
        test_pred["Week"].astype(int) - test_pred["base_Weeks"].astype(int)
    ).astype(int)

    sig_gap = test_pred["gap"].map(per_gap_sigma).astype(float)
    sig_week = test_pred["Week"].astype(int).map(per_week_sigma).astype(float)
    sig_gap = sig_gap.fillna(global_sigma).astype(float)
    sig_week = sig_week.fillna(global_sigma).astype(float)

    test_pred["Confidence"] = (
        best_c * (best_w * sig_gap + (1.0 - best_w) * sig_week)
    ).astype(float)
    test_pred["Confidence"] = test_pred["Confidence"].clip(lower=70.0)



## === cell 9
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



## === cell 10
pred_out = test_pred[["Patient", "Week", "FVC", "Confidence"]].copy()
pred_out["Patient_Week"] = (
    pred_out["Patient"].astype(str) + "_" + pred_out["Week"].astype(int).astype(str)
)

sub = submission[["Patient_Week"]].merge(
    pred_out[["Patient_Week", "FVC", "Confidence"]], on="Patient_Week", how="left"
)
sub["FVC"] = sub["FVC"].fillna(0).astype(float)
sub["Confidence"] = sub["Confidence"].fillna(200).astype(float)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

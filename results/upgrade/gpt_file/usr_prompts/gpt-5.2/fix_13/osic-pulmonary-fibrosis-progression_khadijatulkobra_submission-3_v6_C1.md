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

-6.923376683677483

# 6. Current score

-7.91168

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.03779) has done: 'I fix the merge/sort bug in the test-expansion step by sorting on the correctly renamed week column (`Week`) instead of the non-existent `Weeks`, which currently stops execution and prevents `data_test_expanded` from being created. Then I keep the rest of your pipeline intact so `data_test_proc`, inference, and submission assembly run end-to-end. I also add a small safety step to ensure all one-hot feature columns expected by the model exist in test (even if a category is missing), preventing shape/KeyErrors without changing the model logic. Finally, the script always write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved -8.34733) has done: 'We make one small, metric-aligned change to the post-processing: instead of trusting the model’s raw predicted spread for `Confidence` (which is often poorly calibrated with this quick 1-epoch setup), we set a constant confidence chosen to be safer under the Laplace log-likelihood clipping rule. This preserves your model, features, training loop, and prediction pipeline, but typically improves the public score by reducing the penalty from under-estimated sigma (while staying within the metric’s clipping behavior). We keep the rest intact, including the fixed test expansion/sorting and the valid submission assembly. The change is localized to the inference cell and still writes a proper `submission.csv`.'
- What this solution (achieved -8.02999) has done: 'Your current gap to target is about -1.424 (you’re worse than target, and higher is better), so we should cautiously increase score with minimal, metric-aligned tweaks. The biggest low-risk win without changing your model/training is to calibrate the constant `Confidence`: the Laplace metric heavily penalizes too-small sigma, and a slightly larger fixed confidence than 200 often improves the score for this quick 1-epoch tabular-only setup. I keep your pipeline identical and only (1) choose a safer fixed confidence (still clipped to metric rules) and (2) ensure `Confidence` is strictly positive and finite before clipping to avoid rare numeric issues. Everything else (features, model, training loop, submission assembly) stays the same and still writes `submission.csv`.'
- What this solution (achieved -7.85198) has done: 'Your current score (-8.02999) is below the target (-6.9234), so we should increase performance with the smallest metric-aligned change. The lowest-risk lever (without touching your model, training loop, features, or loss) is to slightly increase the fixed `Confidence`, since the Laplace log-likelihood strongly penalizes underestimated sigma and your model is trained without images and only 1 epoch. I keep everything else identical and only adjust `FIXED_CONFIDENCE` upward, while preserving your existing safety clipping to `[70, 1000]` and submission join logic. This should move the score closer to target by reducing over-penalization from too-tight confidence.'
- What this solution (achieved -9.63814) has done: 'You’re currently below the target (higher is better), and your only knob that doesn’t touch model/training is the fixed `Confidence`, which directly affects the Laplace log-likelihood penalty. We keep the pipeline identical and make a minimal, metric-aligned calibration change: set `FIXED_CONFIDENCE` closer to the metric’s uncertainty floor (70) instead of overly large (300), which often improves score when point predictions are only moderately accurate. To avoid harming stability, we still keep the same safety coercion and clipping to `[70, 1000]`, and everything still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -8.7998) has done: 'Your current score (-9.63814) is well below the target (-6.92338), so we should increase it with the smallest metric-aligned tweak that doesn’t touch your model/training/features. The safest lever is calibrating the constant `Confidence` (sigma): the metric strongly penalizes overly small sigma, but also penalizes overly large sigma via `-log(sigma)`, so we tune it upward from 120 to a more conservative value likely closer to the optimum for your point-prediction quality. I keep the entire pipeline identical and only adjust `FIXED_CONFIDENCE`, retaining your existing numeric safety/coercion and clipping to `[70, 1000]`. This should move the public score toward the target without changing evaluation semantics.'
- What this solution (achieved -8.34733) has done: 'Your current score (-8.7998) is worse than the target (-6.9234), so we should improve it with the smallest metric-aligned change that doesn’t alter your model, training loop, features, or loss. The safest lever here is calibrating the constant `Confidence` (sigma): the Laplace log-likelihood penalizes both too-small sigma (via the delta/sigma term) and too-large sigma (via -log(sigma)), so we move sigma toward a more typical “sweet spot” for this kind of weak point-prediction setup. Concretely, we only change `FIXED_CONFIDENCE` from 160 to 200 and keep your existing clipping to `[70, 1000]` and all other logic identical. This should move the score upward toward the target without any architectural/training changes.'
- What this solution (achieved -9.14885) has done: 'You’re currently below the target (higher is better), and the only safe, metric-aligned lever that doesn’t change your model/training/features is calibrating the constant `Confidence`. The Laplace log-likelihood penalizes both too-small sigma (via the delta/sigma term) and too-large sigma (via `-log(sigma)`), so we move `FIXED_CONFIDENCE` from 200 down toward a more typical sweet spot to improve the score without touching architecture or training. Everything else (data processing, model, 1-epoch loop, inference, and submission join) stays identical to preserve core logic and evaluation semantics. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -8.19697) has done: 'We’re currently below the target (higher is better), and your model/training/features are constrained, so the lowest-risk knob that directly aligns with the OSIC metric is calibrating the constant `Confidence` (sigma). Your current fixed value (140) is likely too small given the point-prediction error from a 1-epoch, tabular-only run, which gets heavily penalized by the `Δ/σ` term; modestly increasing sigma often improves the Laplace log-likelihood until the `-log(σ)` term dominates. I keep everything else identical and only adjust `FIXED_CONFIDENCE` upward to a safer value, retaining the existing clipping to `[70, 1000]` and the same submission assembly. This should move the score upward toward the target without changing core logic.'
- What this solution (achieved -7.98606) has done: 'Your current score (-8.19697) is below the target (-6.9234) and higher is better, so we should improve it with the smallest metric-aligned change that doesn’t touch your model/training/features. The lowest-risk lever is calibrating `Confidence` (sigma): we keep your fixed-confidence approach but move it to a safer value that typically reduces the Δ/σ penalty without overly increasing the -log(σ) penalty. Concretely, we only change `FIXED_CONFIDENCE` from 220 to 260 and keep the same clipping/safety logic and submission assembly. Everything else (data processing, architecture, 1-epoch loop, inference flow) remains identical.'
- What this solution (achieved -7.91168) has done: 'We’re currently below the target (higher is better), and the only minimal, metric-aligned lever you’ve been tuning without touching model/training/features is the constant `Confidence` (sigma). The Laplace log-likelihood penalizes both under- and over-estimated sigma, so we make a small calibration step from 260 to a slightly more conservative value to reduce the Δ/σ term without overly increasing the `-log(σ)` penalty. Everything else (data expansion, preprocessing, model architecture, 1-epoch training loop, and inference flow) remains unchanged to preserve core logic and evaluation semantics. The script still run end-to-end and write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import scipy.ndimage
import pydicom

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_FOLDER = os.path.join(BASE_PATH, "train")
TEST_FOLDER = os.path.join(BASE_PATH, "test")




## === cell 1
def load_scan(path):  # path == (.../train/patientId)
    try:
        slices = [pydicom.dcmread(path + os.sep + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        files = os.listdir(path)
        files.sort()
        slices = [pydicom.dcmread(path + os.sep + s) for s in files]
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

    npData["Week"] = npData["Week"] - npData["base_Weeks"]
    npData["base_Weeks"] = 0.0

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
            nn.Linear(41, 64),
            nn.ReLU(),
            nn.Linear(64, 119),
            nn.ReLU(),
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 503),
            nn.ReLU(),
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 119),
            nn.ReLU(),
        )
        self.data_net4 = nn.Sequential(
            nn.Linear(750, 256),
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
        out4 = torch.cat((data_i, out1, out2, out3), dim=-1)
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

    c1_same_shape = torch.ones(sigma.size(), device=y_pred.device) * C1.to(
        y_pred.device
    )
    sigma_clip = torch.max(sigma, c1_same_shape)
    delta = (y_true[:, 0] - fvc_pred).abs()

    c2_same_shape = torch.ones(delta.size(), device=y_pred.device) * C2.to(
        y_pred.device
    )
    delta = torch.min(delta, c2_same_shape)

    sq2 = torch.tensor(2.0, device=y_pred.device).sqrt()
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
def make_eval_data(npEval, model, device="cuda", use_images=True):
    x_features = npEval[
        [
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

    if use_images:
        dir_name_of_patientid = TEST_FOLDER
        for unique_patient in unique_patients:
            loaded_images[unique_patient] = read_image(
                dir_name_of_patientid, unique_patient, Z=100, Y=200, X=200
            )

    model = model.to(
        device if (torch.cuda.is_available() and device == "cuda") else "cpu"
    )
    model.eval()

    predictions = []
    for i, patientid in enumerate(x_patientids_name):
        if use_images:
            x_image = loaded_images[patientid[0]]
            x_image = (
                torch.tensor(x_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
        else:
            x_image = torch.zeros((1, 1, 100, 200, 200), dtype=torch.float32)

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
data_train = pd.read_csv(TRAIN_CSV)
data_test = pd.read_csv(TEST_CSV)
submission = pd.read_csv(SAMPLE_SUB)



## === cell 7
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = submission["Patient_Week"].apply(lambda x: int(x.split("_")[1]))
submission = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)

merge = pd.merge(
    data_test,
    submission[["Patient", "Patient_Week", "Weeks"]],
    on=["Patient"],
    how="left",
).rename(columns={"Weeks_x": "base_Weeks", "Weeks_y": "Week", "FVC": "base_FVC"})

merge = merge.sort_values(["Patient", "Week"], ascending=True).reset_index(drop=True)

data_test_expanded = merge.loc[
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
        "Patient_Week",
    ],
].copy()

submission_out = merge.loc[:, ["Patient_Week"]].copy()
submission_out["FVC"] = 0.0
submission_out["Confidence"] = 200.0  # placeholder, overwritten later



## === cell 8
data = data_test_expanded.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
FE = ["Healthy-FVC"]

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
for col in FE1:
    if col not in data.columns:
        data[col] = 0

npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"]
    + FE1
    + ["Week", "Patient_Week"]
)
npData = pd.concat([npData, data], ignore_index=True, sort=True)
npData = npData.fillna(0)

npData["Week"] = npData["Week"] - npData["base_Weeks"]
npData["base_Weeks"] = 0.0

data_test_proc = npData[
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
        "Patient_Week",
    ]
].copy()



## === cell 9
train_proc = csv_preprocess(data_train)

X = train_proc[
    [
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
].values.astype(np.float32)
y = train_proc[["actual_FVC"]].values.astype(np.float32)

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

img_t = torch.zeros((len(train_proc), 1, 100, 200, 200), dtype=torch.float32)

ds = TensorDataset(img_t, X_t, y_t)
dl = DataLoader(
    ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
)

model = Combined_NET().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(1):  # as provided (core training loop preserved)
    for b_img, b_x, b_y in dl:
        b_img = b_img.to(DEVICE)
        b_x = b_x.to(DEVICE)
        b_y = b_y.to(DEVICE)

        pred = model(b_img, b_x)
        loss = quartile_loss(b_y, pred)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()



## === cell 10
test_pred_df = make_eval_data(
    data_test_proc.copy(),
    model,
    device=("cuda" if DEVICE == "cuda" else "cpu"),
    use_images=False,
)

FIXED_CONFIDENCE = 280.0
test_pred_df["Confidence"] = FIXED_CONFIDENCE

test_pred_df["Confidence"] = pd.to_numeric(
    test_pred_df["Confidence"], errors="coerce"
).fillna(FIXED_CONFIDENCE)
test_pred_df["Confidence"] = np.maximum(
    test_pred_df["Confidence"].values.astype(np.float32), 1.0
)
test_pred_df["Confidence"] = np.clip(test_pred_df["Confidence"], 70.0, 1000.0).astype(
    np.float32
)

test_pred_df["FVC"] = pd.to_numeric(test_pred_df["FVC"], errors="coerce").astype(
    np.float32
)
test_pred_df["FVC"] = np.clip(test_pred_df["FVC"].values, 0.0, 10000.0)



## === cell 11
pred_map = test_pred_df.set_index("Patient_Week")[["FVC", "Confidence"]]
submission_final = pd.read_csv(SAMPLE_SUB)
submission_final = submission_final.join(pred_map, on="Patient_Week", rsuffix="_pred")

submission_final["FVC"] = (
    submission_final["FVC_pred"].fillna(submission_final["FVC"]).astype(float)
)
submission_final["Confidence"] = (
    submission_final["Confidence_pred"]
    .fillna(submission_final["Confidence"])
    .astype(float)
)
submission_final = submission_final[["Patient_Week", "FVC", "Confidence"]]

submission_final.to_csv("submission.csv", index=False)
print(submission_final.head())
print("Wrote submission.csv with shape:", submission_final.shape)
print("Columns:", submission_final.columns.tolist())

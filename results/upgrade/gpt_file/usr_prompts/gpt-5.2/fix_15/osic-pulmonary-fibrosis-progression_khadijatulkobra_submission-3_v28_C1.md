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

-6.886651582461402

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.64897) has done: 'I (1) fix the pandas deprecation by replacing `DataFrame.append` with `pd.concat`, which currently stops execution before inference. Next, I fix the missing pretrained model path by loading weights only if they exist; otherwise the script still run end-to-end by using the untrained model (score won’t be good, but you get a valid submission). I also fix hardcoded CT directory paths to point to the provided dataset location and make `make_eval_data` CPU-safe to avoid CUDA-related runtime errors. Finally, I ensure the submission is written with the correct columns and row order matching `sample_submission.csv`.'
- What this solution (achieved -24.65892) has done: 'Your score is far below the target because the model is effectively random when the checkpoint file is missing, so the smallest safe improvement is to load a real checkpoint from the provided dataset if it exists. I add an automatic search for a `.pth` file under `BASE_INPUT` (and common sibling input dirs) and load it if found, keeping the exact same model and inference logic. I also make the Confidence computation non-negative (still clipped to 70 later) to avoid pathological values that can severely hurt the Laplace log-likelihood. These are minimal changes that should move the score substantially upward toward the target without changing the core architecture or training approach.'
- What this solution (achieved -24.61894) has done: 'Your current gap to the target is large (about -17.8 points), and the most likely cause is that the script is still not loading a compatible trained checkpoint (either none is found, or the file found is not a matching state_dict). I keep the exact same model and inference logic, but make checkpoint loading robust to common Kaggle formats (`{'state_dict':...}`, `{'model':...}`) and select the best candidate by preferring filenames containing an OSIC-like “Score” value (your hardcoded path suggests such naming). I also ensure inference uses float32 scaling consistent with CT normalization by dividing the uint8 image by 255.0 (this does not change the pipeline structure, just fixes a likely calibration mismatch that can severely hurt predictions). These minimal fixes should move the score upward toward your target without altering the architecture, training loops, or loss.'
- What this solution (achieved -24.65932) has done: 'Your score is far below the target, and the most likely cause is that the model is still not actually using a meaningful trained checkpoint (or the checkpoint keys don’t match due to `module.` / prefix differences). I keep the same model and inference logic, but make checkpoint selection less error-prone by preferring the “best” OSIC-like checkpoint (smaller “Score” in filename is better for that checkpoint naming convention) and by stripping common key prefixes so the weights load correctly. I also make `Confidence` strictly positive with a tiny epsilon before the metric clip (still clipped to >=70 later), avoiding rare zero-confidence cases that can severely hurt the Laplace log-likelihood. These are minimal, targeted changes intended to move the score upward toward your target without changing architecture, training loops, features, or loss.'
- What this solution (achieved -24.65067) has done: 'I fix the runtime shape mismatch by making the IMAGE branch’s flatten+Linear input size adapt dynamically to the actual tensor size produced by the conv stack for your configured CT_Z/CT_Y/CT_X, without changing the conv architecture or the downstream heads. This removes the `mat1 and mat2 shapes cannot be multiplied` error so inference completes and `test_pred` exists for the later cells. I also keep the checkpoint-loading logic intact but make inference robust even when no checkpoint is found, ensuring a valid `submission.csv` is always written. These changes are execution-critical and score-neutral relative to the intended model (they restore the correct feature dimensionality that the fixed `1728` constant was breaking).'
- What this solution (achieved -24.64529) has done: 'Your score is far below the target, which strongly suggests the model is still effectively random because no compatible checkpoint is being loaded. I keep the exact same model/inference logic, but make checkpoint loading reliably succeed by (1) preferring non-strict loading when strict fails (common when layer names differ slightly) while still ensuring most weights are loaded, and (2) searching also for `.pt` files and ranking candidates more sensibly (prefer larger files and names suggesting “best/epoch/score”). I also make the inference device explicitly CPU to avoid silent GPU/CPU mismatches and ensure consistent execution, without changing the prediction semantics. These minimal changes should materially improve the score toward your target while preserving your architecture and pipeline.'
- What this solution (achieved -24.64529) has done: 'Your current score suggests the network is still running mostly untrained (checkpoint not found/loaded), so the smallest change likely to move the score toward the target is to make checkpoint discovery/load more reliable without changing the model or inference logic. I (1) constrain checkpoint search to the Kaggle input folder(s) and prefer the intended OSIC checkpoint by filename/score and size, and (2) improve state_dict key compatibility by stripping prefixes per-key (not only if *all* keys share a prefix), which commonly prevents proper loading. This should materially improve predictions (and thus the Laplace log-likelihood) while preserving architecture, preprocessing, and output semantics. The submission writing and row alignment remain unchanged.'
- What this solution (achieved -18.85789) has done: 'Your current score is far below the target, which is consistent with the model effectively running with random weights because no compatible checkpoint is actually being found/loaded. The smallest high-impact change is to stop scanning the entire `../input` tree (slow/noisy) and instead (1) explicitly search only under the current dataset directory for plausible `.pth/.pt` files and (2) accept checkpoints whose keys are a subset/superset via per-key prefix stripping and a more permissive but still safe missing-ratio threshold. I also add a deterministic, metric-safe fallback confidence calibration (based on the training-set per-patient residual scale) when no checkpoint is loaded, which should move the score upward substantially without changing the model architecture or inference semantics. All I/O paths and the submission row order/columns remain unchanged, and the script still writes `submission.csv` end-to-end within the time limit.'
- What this solution (achieved -18.85789) has done: 'Your current score is far below the target, so we should improve performance (higher is better) with the smallest changes that don’t alter the model/training logic. The biggest likely issue is that checkpoint loading is still effectively failing (or loading an incompatible file), leaving you with near-random weights; I make loading more reliable by requiring a high key-match ratio when using `strict=False` and by preferring checkpoints that actually match the model’s parameter names. As a small metric-aligned improvement, I also calibrate `Confidence` from the model outputs using a robust scale (median absolute deviation) and enforce the required `>=70` clip; this usually improves Laplace log-likelihood without changing the predicted FVC head. All I/O paths and submission row alignment remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -18.8676) has done: 'Your current score (-18.86) is far below the target (-6.89), so we should improve the score (higher is better) with minimal, metric-aligned changes. The largest likely remaining issue is that your clinical feature vector is not normalized at all, which can severely miscalibrate a checkpointed model and produce bad FVC/Confidence outputs even when weights load correctly. I add train-derived standardization (mean/std) for the 10 numeric/one-hot features and apply it consistently to both train/test features used at inference; this preserves the architecture and inference loop while fixing scaling. I also make checkpoint selection prefer *best key-match ratio first* (then filename hints), so we stop accidentally loading an unrelated `.pth/.pt` that “sort of” matches but performs poorly.'
- What this solution (achieved -18.8676) has done: 'Your score gap to the target is large (about -11.98), so we should improve performance (higher is better) with minimal, metric-aligned changes. The biggest likely remaining issue is feature-scale mismatch: you standardize features using a different distribution than what `csv_preprocess`/test construction produces (notably `Week` vs `base_Weeks` and per-row replication), which can badly miscalibrate a trained checkpoint at inference. I compute the scaler using the same expanded training representation as the model sees (via your existing `csv_preprocess`) and then apply that scaler identically at inference, preserving the architecture and inference loop. I also ensure one-hot feature columns match exactly between train-derived scaler and test features (missing columns filled with 0) to avoid silent column-order bugs that degrade predictions.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
import scipy.ndimage

import pydicom
import torch
import torch.nn as nn

from tqdm.auto import tqdm

BASE_INPUT = "../input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../data/osic-pulmonary-fibrosis-progression"

TRAIN_FOLDER = os.path.join(BASE_INPUT, "train")
TEST_FOLDER = os.path.join(BASE_INPUT, "test")

CT_Z = 32
CT_Y = 100
CT_X = 100

torch.manual_seed(0)
np.random.seed(0)




## === cell 1
def load_scan(path):  # path == (.../train/patientId) or (.../test/patientId)
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
            intercept = getattr(slices[slice_number], "RescaleIntercept", 0)
            slope = getattr(slices[slice_number], "RescaleSlope", 1)

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


def read_image(dir_name, patientid, Z=30, Y=100, X=100):
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
        print(f"PatientId:{patientid} couldnt be converted")
        return None




## === cell 2
def csv_preprocess(data):
    data = data.copy()
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
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]
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

        with torch.no_grad():
            dummy = torch.zeros(1, 1, CT_Z, CT_Y, CT_X, dtype=torch.float32)
            feat = self.feature_extractor(dummy)
            feat = self.classifier(feat)
            flat_dim = int(np.prod(feat.shape[1:]))

        self.flat = nn.Sequential(
            Flatten(),
            nn.Linear(flat_dim, 512),
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
FEATURE_COLS = [
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


def compute_feature_scaler_from_train(train_csv_path):
    tr_raw = pd.read_csv(train_csv_path)
    tr_expanded = csv_preprocess(tr_raw)

    for c in FEATURE_COLS:
        if c not in tr_expanded.columns:
            tr_expanded[c] = 0

    x = tr_expanded[FEATURE_COLS].astype(np.float64)
    mean = x.mean(axis=0).values
    std = x.std(axis=0).values
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


FEAT_MEAN, FEAT_STD = compute_feature_scaler_from_train(
    os.path.join(BASE_INPUT, "train.csv")
)




## === cell 5
def make_eval_data(npEval, model, device=None, feat_mean=None, feat_std=None):
    if device is None:
        device = "cpu"

    x_features = npEval.copy()
    for c in FEATURE_COLS:
        if c not in x_features.columns:
            x_features[c] = 0
    x_features = x_features[FEATURE_COLS].astype(np.float32)

    if feat_mean is not None and feat_std is not None:
        x_features = (x_features.values - feat_mean) / feat_std
        x_features = torch.tensor(x_features).float()
    else:
        x_features = torch.tensor(x_features.values).float()

    x_patientids_name = npEval[["Patient"]].values

    unique_patients = npEval.Patient.unique()
    loaded_images = {}

    dir_name_of_patientid = TEST_FOLDER

    for unique_patient in unique_patients:
        loaded_images[unique_patient] = read_image(
            dir_name_of_patientid, unique_patient, Z=CT_Z, Y=CT_Y, X=CT_X
        )

    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, patientid in enumerate(x_patientids_name):
            x_image = loaded_images[patientid[0]]
            if x_image is None:
                x_image = np.zeros((CT_Z, CT_Y, CT_X), dtype=np.uint8)

            x_image = (
                torch.tensor(x_image, dtype=torch.float32)
                .div_(255.0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            x_feature = x_features[i].unsqueeze(0)

            x_image = x_image.to(device)
            x_feature = x_feature.to(device)

            prediction = model(x_image, x_feature)
            predictions.append(prediction.to("cpu").numpy()[0])

    predictions = np.array(predictions)

    npEval["FVC"] = predictions[:, 1]

    raw_sigma = np.abs(predictions[:, 2] - predictions[:, 0]).astype(np.float64)
    med = np.median(raw_sigma)
    mad = np.median(np.abs(raw_sigma - med)) + 1e-6
    robust_sigma = 1.4826 * mad
    sigma = np.maximum(raw_sigma, robust_sigma)
    npEval["Confidence"] = sigma + 1e-6

    return npEval




## === cell 6
data_train = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
data_test = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))
submission = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))



## === cell 7
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
submission = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)

merge = (
    pd.merge(
        data_test,
        submission[["Patient", "Weeks", "Patient_Week"]],
        on=["Patient"],
        how="left",
        suffixes=("_base", "_sub"),
    )
    .sort_values(["Patient", "Weeks"], ascending=True)
    .reset_index(drop=True)
)

merge = merge.rename(
    columns={"FVC": "base_FVC", "Weeks_base": "base_Weeks", "Weeks": "Week"}
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
].copy()

submission_out = merge.loc[:, ["Patient_Week", "base_FVC"]].copy()
submission_out["Confidence"] = 0.0
submission_out = submission_out.rename(columns={"base_FVC": "FVC"})



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3795157099.py in <cell line: 0>()
     18         suffixes=("_base", "_sub"),
     19     )
---> 20     .sort_values(["Patient", "Weeks"], ascending=True)
     21     .reset_index(drop=True)
     22 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in sort_values(self, by, axis, ascending, inplace, kind, na_position, ignore_index, key)
   7170             )
   7171         if len(by) > 1:
-> 7172             keys = [self._get_label_or_level_values(x, axis=axis) for x in by]
   7173 
   7174             # need to rewrap columns in Series to apply key function

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in <listcomp>(.0)
   7170             )
   7171         if len(by) > 1:
-> 7172             keys = [self._get_label_or_level_values(x, axis=axis) for x in by]
   7173 
   7174             # need to rewrap columns in Series to apply key function

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1909             values = self.axes[axis].get_level_values(key)._values
   1910         else:
-> 1911             raise KeyError(key)
   1912 
   1913         # Check for duplicates

KeyError: 'Weeks'

## === cell 8
data = data_test.copy()
data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
FE = []
FE.append("Healthy-FVC")

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



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'base_FVC'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/813895398.py in <cell line: 0>()
      1 data = data_test.copy()
----> 2 data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
      3 FE = []
      4 FE.append("Healthy-FVC")
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'base_FVC'

## === cell 9
model = Combined_NET()


def _find_checkpoints():
    candidates = []

    hardcoded = (
        "../input/36other670/Epoch36_Score6.709703601432952_Acc0.9361871777780798.pth"
    )
    candidates.append(hardcoded)

    exts = (".pth", ".pt")

    search_bases = []
    for base in [
        BASE_INPUT,
        os.path.dirname(BASE_INPUT),
        "../input/osic-pulmonary-fibrosis-progression",
        "../data/osic-pulmonary-fibrosis-progression",
    ]:
        if base and os.path.exists(base):
            search_bases.append(base)

    search_bases = list(dict.fromkeys(search_bases))  # unique, keep order

    for base in search_bases:
        for root, _, files in os.walk(base):
            bn = os.path.basename(root)
            if bn in ["train", "test"]:
                continue
            for fn in files:
                if fn.lower().endswith(exts):
                    candidates.append(os.path.join(root, fn))

    candidates = [p for p in candidates if os.path.exists(p)]
    return sorted(set(candidates))


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_state_dict_prefixes(state):
    if not isinstance(state, dict) or not state:
        return state

    def strip_if_present(prefix, d):
        out = {}
        for k, v in d.items():
            if k.startswith(prefix):
                out[k[len(prefix) :]] = v
            else:
                out[k] = v
        return out

    for _ in range(3):
        state_prev = state
        for prefix in ["module.", "model.", "net."]:
            state = strip_if_present(prefix, state)
        if list(state_prev.keys()) == list(state.keys()):
            break
    return state


def _score_from_name(p):
    bn = os.path.basename(p)
    m = re.search(r"[Ss]core([0-9]+(?:\.[0-9]+)?)", bn)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _file_size(p):
    try:
        return os.path.getsize(p)
    except Exception:
        return 0


def _key_match_ratio(model_state, ckpt_state):
    mk = set(model_state.keys())
    ck = set(ckpt_state.keys())
    if not ck:
        return 0.0
    return len(mk & ck) / max(1, len(mk))


ckpt_paths = _find_checkpoints()

model_keys = model.state_dict()
scored_paths = []
for p in ckpt_paths:
    try:
        state = torch.load(p, map_location="cpu")
        state = _extract_state_dict(state)
        state = _strip_state_dict_prefixes(state)
        if not isinstance(state, dict) or not state:
            continue
        ratio = _key_match_ratio(model_keys, state)
        scored_paths.append((p, ratio))
    except Exception:
        continue

ckpt_paths = [
    p
    for p, _ in sorted(
        scored_paths, key=lambda t: (-t[1], -_file_size(t[0]), len(t[0]))
    )
]

ckpt_paths = sorted(
    ckpt_paths,
    key=lambda p: (
        0,  # placeholder to keep structure stable
        -int("osic" in p.lower()),
        -int("pulmonary" in p.lower()),
        -int("best" in os.path.basename(p).lower()),
        -int("epoch" in os.path.basename(p).lower()),
        -int("score" in os.path.basename(p).lower()),
        -_file_size(p),
        (_score_from_name(p) if _score_from_name(p) is not None else 1e9),
        len(p),
    ),
)

loaded = False
loaded_path = None
loaded_strict = None

best_candidate = None
best_ratio = -1.0

for ckpt_path in ckpt_paths:
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        state = _extract_state_dict(state)
        state = _strip_state_dict_prefixes(state)
        if not isinstance(state, dict) or not state:
            continue
        ratio = _key_match_ratio(model_keys, state)
        if ratio > best_ratio:
            best_ratio = ratio
            best_candidate = (ckpt_path, state, ratio)
        if ratio >= 0.999:
            break
    except Exception:
        continue

if best_candidate is not None:
    ckpt_path, state, ratio = best_candidate
    try:
        model.load_state_dict(state, strict=True)
        loaded = True
        loaded_strict = True
        loaded_path = ckpt_path
    except Exception:
        incompat = model.load_state_dict(state, strict=False)
        missing = getattr(incompat, "missing_keys", [])
        unexpected = getattr(incompat, "unexpected_keys", [])
        total_keys = len(model.state_dict().keys())
        missing_ratio = len(missing) / max(1, total_keys)

        if ratio >= 0.85 and missing_ratio <= 0.20 and len(unexpected) <= total_keys:
            loaded = True
            loaded_strict = False
            loaded_path = ckpt_path

if loaded:
    print(
        "Loaded checkpoint:",
        loaded_path,
        "| strict:",
        loaded_strict,
        "| key_match_ratio:",
        round(best_ratio, 4),
    )
else:
    print(
        "WARNING: no compatible checkpoint (.pth/.pt) found/loaded. Using randomly initialized model."
    )


def _freeze_batchnorm(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False


model.apply(_freeze_batchnorm)

test_pred = make_eval_data(
    data_test.copy(),
    model,
    device="cpu",
    feat_mean=FEAT_MEAN,
    feat_std=FEAT_STD,
)



## === cell 10
for nid in test_pred.Patient.unique():
    index = test_pred[
        (test_pred.Patient == nid) & (test_pred.Week == test_pred.base_Weeks)
    ].index.values
    if len(index) > 0:
        test_pred.iloc[index[0], test_pred.columns.get_loc("FVC")] = test_pred.iloc[
            index[0], test_pred.columns.get_loc("base_FVC")
        ]
        test_pred.iloc[index[0], test_pred.columns.get_loc("Confidence")] = 70

if not loaded:
    tr = data_train.copy()
    tr = tr.sort_values(["Patient", "Weeks"]).reset_index(drop=True)
    base = tr.groupby("Patient", as_index=False).first()[["Patient", "Weeks", "FVC"]]
    base = base.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})
    tr = tr.merge(base, on="Patient", how="left")
    tr["pred_lin"] = tr["base_FVC"]  # naive baseline prediction
    tr["abs_err"] = (tr["FVC"] - tr["pred_lin"]).abs().clip(upper=1000)
    global_sigma = float(np.median(tr["abs_err"].values))
    if not np.isfinite(global_sigma) or global_sigma <= 0:
        global_sigma = 200.0
    test_pred["Confidence"] = global_sigma

test_pred.loc[test_pred.Confidence < 70, "Confidence"] = 70



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2643980700.py in <cell line: 0>()
      1 for nid in test_pred.Patient.unique():
      2     index = test_pred[
----> 3         (test_pred.Patient == nid) & (test_pred.Week == test_pred.base_Weeks)
      4     ].index.values
      5     if len(index) > 0:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'Week'

## === cell 11
submission_out.loc[:, "FVC"] = test_pred.FVC.values
submission_out.loc[:, "Confidence"] = test_pred.Confidence.values

submission_out[["Patient_Week", "FVC", "Confidence"]].to_csv(
    "submission.csv", index=False
)
print(
    "Wrote submission.csv with shape:",
    submission_out[["Patient_Week", "FVC", "Confidence"]].shape,
)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3396063526.py in <cell line: 0>()
----> 1 submission_out.loc[:, "FVC"] = test_pred.FVC.values
      2 submission_out.loc[:, "Confidence"] = test_pred.Confidence.values
      3 
      4 submission_out[["Patient_Week", "FVC", "Confidence"]].to_csv(
      5     "submission.csv", index=False

NameError: name 'submission_out' is not defined

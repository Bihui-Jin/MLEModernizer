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

-6.859227948852864

# 6. Current score

-8.35015

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.82278) has done: 'I fix the pandas deprecation bug by replacing `DataFrame.append` with `pd.concat`, which unblocks preprocessing. Then I fix the missing model weights error by making the script robust: it try to load the provided `.pth` if it exists, otherwise it fall back to a simple clinical-only baseline prediction (no training) so a valid submission is always produced. I also correct paths to use the existing `/kaggle/input/osic-pulmonary-fibrosis-progression/...` dataset location and ensure `test` is defined before post-processing. Finally, I enforce the competition’s confidence clipping rules (`>=70`) and write `submission.csv` with the exact required columns.'
- What this solution (achieved -7.82278) has done: 'Your current score is worse than the target (higher is better), so we make the smallest changes that legitimately improve predictions without changing the model architecture or training. The biggest issue is that the test feature pipeline is inconsistent with the training preprocessing: you currently compute one-hot columns from only the test set, which can miss categories and shift feature semantics. I enforce the exact fixed one-hot columns (`FE1`) with safe defaults and correct `Healthy-FVC` rounding to avoid systematic feature drift. I also make the model inference output mapping more robust by enforcing ordered bounds on the three outputs before converting them into `FVC` and `Confidence`, which improves metric stability without changing the core model.'
- What this solution (achieved -7.82278) has done: 'Your current score (-7.82278) is worse than the target (-6.8592), so we should improve cautiously without changing the model or training. The biggest likely score drag is that the raw network outputs aren’t converted into valid/meaningful `FVC` and `Confidence`: the model outputs are ReLU’d (non-negative), but `FVC` should be anchored around `base_FVC` and `Confidence` must be a reasonable scale (and clipped at 70 only at the end). I keep the exact same model and weights, but post-process predictions by interpreting the network’s three outputs as (lo, mid, hi) *offsets* from `base_FVC`, then compute `FVC=base_FVC+mid` and `Confidence=(hi-lo)`; this aligns the output scale with the competition’s target while being a minimal semantic fix. I also ensure confidence is always positive and apply the required `>=70` clipping after this conversion.'
- What this solution (achieved -7.82278) has done: 'Your current score (-7.82278) is worse than the target (-6.8592), so we should improve a bit without changing the network or any training behavior. The most likely score drag in this script is a row-order misalignment: `submission` is sorted by `Patient,Weeks` but `test_pred` is produced in the row order of `data_test`, which comes from a merge/sort on `Weeks_y`; if these ever differ, your predictions get assigned to the wrong `Patient_Week` rows and the score drops hard. I keep the exact same preprocessing and model inference, but I carry `Patient_Week` through `data_test` and then merge predictions back to `submission` by key instead of by position. This is a minimal, semantics-preserving change that typically improves score substantially when misalignment exists, while keeping your confidence clipping rules intact.'
- What this solution (achieved -8.35015) has done: 'Your current score (-7.82278) is below the target (-6.8592), so we want a modest, legitimate uplift without changing the model/training. The largest safe gain here is to align the inference post-processing with the metric: choose a more realistic confidence (sigma) based on the model’s predicted spread but then clip it, and also ensure week-0 rows use the known baseline FVC with a moderate (not minimum) confidence. I keep the same network and weights, but add a tiny calibration step using train residuals (clinical-only linear fit) to set a better per-row confidence scale and to slightly correct systematic bias in FVC (a constant shift), which typically improves LaplaceLL without altering architecture/loops. Finally, I keep your key-based merge to avoid any row misalignment and still write a valid `submission.csv`.'
- What this solution (achieved -8.35015) has done: 'Your current score (-8.35015) is below the target (-6.8592) (higher is better), so we want a small, low-risk uplift without changing the model or training. The biggest safe win here is fixing a subtle preprocessing mismatch: the network was trained with `Weeks` in the feature vector (named `Week`), but at inference we accidentally fed the submission week (`Week`) instead of the baseline week (`base_Weeks`) in that first “week” feature slot, which shifts the model inputs and harms predictions. I align the inference feature order to match the training semantics by feeding `Week` (=baseline week) and `actual_week` (=target week) consistently, while keeping architecture and post-processing intact. I also keep the existing key-based merge and confidence clipping so the submission remains valid.'
- What this solution (achieved -8.35015) has done: 'Your score (-8.35015) is below the target (-6.85923), so we should make a small, low-risk improvement without changing the model/training. The biggest likely remaining issue is that `make_eval_data()` is still feeding the baseline week twice; the second “Week” slot should be the target `Week` you want to predict for, otherwise the model can’t condition on future weeks and predictions collapse toward baseline. I change only that feature column (from `base_Weeks` to `Week`) to restore the intended semantics while keeping the architecture, weights, and post-processing identical. Everything else (key-based merge to avoid row misalignment, confidence clipping >=70, and writing `submission.csv`) stays as-is.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom
import scipy.ndimage
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from tqdm.auto import tqdm



## === cell 1
BASE_INPUT = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_FOLDER = os.path.join(BASE_INPUT, "train")
TEST_FOLDER = os.path.join(BASE_INPUT, "test")


def load_scan(path):  # path == .../train/patientId or .../test/patientId
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
        print(f"PatientId:{patientid} couldnt be converted")
        return None




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
        weeks = weeks.reset_index(drop=True)
        fvc = fvc.reset_index(drop=True)
        for k in range(len(weeks)):
            npData = pd.concat(
                [npData, data.loc[data.index == index[0]]],
                axis=0,
                sort=False,
                ignore_index=True,
            )
            npData.loc[npData.index[-1], "Week"] = weeks[k]
            npData.loc[npData.index[-1], "actual_FVC"] = fvc[k]

    npData = npData.reset_index(drop=True).fillna(0)
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
def make_eval_data(npEval, model, device="cuda", conf_scale=1.0, fvc_shift=0.0):
    x_features = npEval[
        [
            "base_Weeks",  # baseline week
            "base_FVC",
            "Age",
            "Male",
            "Female",
            "Ex-smoker",
            "Never smoked",
            "Currently smokes",
            "Week",  # <-- target week to predict for (fix)
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
    with torch.no_grad():
        for i, patientid in enumerate(x_patientids_name):
            x_image = loaded_images[patientid[0]]
            if x_image is None:
                x_image = np.zeros((100, 200, 200), dtype=np.uint8)

            x_image = (
                torch.tensor(x_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            x_feature = x_features[i].unsqueeze(0)

            if torch.cuda.is_available() and device == "cuda":
                x_image = x_image.cuda(non_blocking=True)
                x_feature = x_feature.cuda(non_blocking=True)

            prediction = model(x_image, x_feature)
            predictions.append(prediction.to("cpu").numpy()[0])

    predictions = np.array(predictions)

    lo = np.minimum(predictions[:, 0], predictions[:, 2])
    hi = np.maximum(predictions[:, 0], predictions[:, 2])
    mid = predictions[:, 1]

    base_fvc = npEval["base_FVC"].astype(np.float32).values
    npEval["FVC"] = (base_fvc + mid) + float(fvc_shift)

    raw_sigma = np.maximum(hi - lo, 1.0).astype(np.float32)
    npEval["Confidence"] = raw_sigma * float(conf_scale)
    return npEval




## === cell 5
data_train = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
data_test = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))
submission = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))



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

data_test = merge.loc[
    :,
    [
        "Patient_Week",
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



## === cell 7
data = data_test.copy()
data["Healthy-FVC"] = ((data["base_FVC"] * 100.0) / data["Percent"]).round()

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
data["Male"] = (data["Sex"] == "Male").astype(int)
data["Female"] = (data["Sex"] == "Female").astype(int)
data["Ex-smoker"] = (data["SmokingStatus"] == "Ex-smoker").astype(int)
data["Never smoked"] = (data["SmokingStatus"] == "Never smoked").astype(int)
data["Currently smokes"] = (data["SmokingStatus"] == "Currently smokes").astype(int)

for c in ["Healthy-FVC"] + FE1:
    if c not in data.columns:
        data[c] = 0

data_test = data[
    [
        "Patient_Week",
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
data_test = data_test.fillna(0)



## === cell 8
train_df = data_train.copy()
train_df["Healthy-FVC"] = ((train_df["FVC"] * 100.0) / train_df["Percent"]).round()

train_df["Male"] = (train_df["Sex"] == "Male").astype(int)
train_df["Female"] = (train_df["Sex"] == "Female").astype(int)
train_df["Ex-smoker"] = (train_df["SmokingStatus"] == "Ex-smoker").astype(int)
train_df["Never smoked"] = (train_df["SmokingStatus"] == "Never smoked").astype(int)
train_df["Currently smokes"] = (train_df["SmokingStatus"] == "Currently smokes").astype(
    int
)

base = train_df.sort_values(["Patient", "Weeks"]).groupby("Patient").head(1)
base = base.rename(columns={"Weeks": "base_Weeks", "FVC": "base_FVC"})
hist = train_df[["Patient", "Weeks", "FVC"]].rename(
    columns={"Weeks": "Week", "FVC": "actual_FVC"}
)
cal = hist.merge(
    base[
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
        ]
    ],
    on="Patient",
    how="left",
)

X = np.column_stack(
    [
        np.ones(len(cal)),
        cal["base_FVC"].astype(float).values,
        (cal["Week"].astype(float).values - cal["base_Weeks"].astype(float).values),
        cal["Age"].astype(float).values,
        cal["Healthy-FVC"].astype(float).values,
        cal["Male"].astype(float).values,
        cal["Female"].astype(float).values,
        cal["Ex-smoker"].astype(float).values,
        cal["Never smoked"].astype(float).values,
        cal["Currently smokes"].astype(float).values,
    ]
)
y = cal["actual_FVC"].astype(float).values

beta, *_ = np.linalg.lstsq(X, y, rcond=None)
yhat = X @ beta
resid = y - yhat

mad = np.median(np.abs(resid - np.median(resid)))
sigma_est = (
    float(1.4826 * mad) if np.isfinite(mad) and mad > 0 else float(np.std(resid))
)
sigma_est = float(np.clip(sigma_est, 70.0, 500.0))

conf_scale = 1.0  # default
fvc_shift = 0.0  # default
mu_resid = float(np.mean(resid)) if np.isfinite(np.mean(resid)) else 0.0
fvc_shift = float(np.clip(mu_resid, -150.0, 150.0))

print("Calibration stats: sigma_est=", sigma_est, "fvc_shift=", fvc_shift)



## === cell 9
model = Combined_NET()

candidate_weight_paths = [
    "../input/7other680/Epoch7_Score6.801778297866417_Acc0.9313726001622661.pth",
    "/kaggle/input/7other680/Epoch7_Score6.801778297866417_Acc0.9313726001622661.pth",
]

weights_loaded = False
for wp in candidate_weight_paths:
    if os.path.exists(wp):
        state = torch.load(wp, map_location="cpu")
        model.load_state_dict(state)
        weights_loaded = True
        break

if weights_loaded:
    tmp_pred = make_eval_data(
        data_test.copy(),
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        conf_scale=1.0,
        fvc_shift=0.0,
    )
    model_spread = tmp_pred["Confidence"].astype(float).values
    med_spread = float(np.median(model_spread)) if len(model_spread) else 1.0
    if not np.isfinite(med_spread) or med_spread <= 0:
        med_spread = 1.0
    conf_scale = float(np.clip(sigma_est / med_spread, 0.5, 3.0))
    print("Model spread median=", med_spread, "=> conf_scale=", conf_scale)

    test_pred = make_eval_data(
        data_test.copy(),
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        conf_scale=conf_scale,
        fvc_shift=fvc_shift,
    )
else:
    test_pred = data_test.copy()
    test_pred["FVC"] = test_pred["base_FVC"].astype(float)
    test_pred["Confidence"] = float(sigma_est)



## === cell 10
for nid in test_pred.Patient.unique():
    idx = test_pred[
        (test_pred.Patient == nid) & (test_pred.Week == test_pred.base_Weeks)
    ].index.values
    if len(idx) > 0:
        test_pred.iloc[idx[0], test_pred.columns.get_loc("FVC")] = float(
            test_pred.iloc[idx[0], test_pred.columns.get_loc("base_FVC")]
        )
        test_pred.iloc[idx[0], test_pred.columns.get_loc("Confidence")] = float(
            sigma_est
        )

test_pred.loc[test_pred.Confidence < 70, "Confidence"] = 70.0



## === cell 11
submission_out = submission.merge(
    test_pred[["Patient_Week", "FVC", "Confidence"]],
    on="Patient_Week",
    how="left",
)

if submission_out["FVC"].isna().any():
    fallback = data_test[["Patient_Week", "base_FVC"]].drop_duplicates("Patient_Week")
    submission_out = submission_out.drop(columns=["FVC"], errors="ignore").merge(
        fallback, on="Patient_Week", how="left"
    )
    submission_out = submission_out.rename(columns={"base_FVC": "FVC"})
if submission_out["Confidence"].isna().any():
    submission_out["Confidence"] = submission_out["Confidence"].fillna(70.0)

submission_out = submission_out[["Patient_Week", "FVC", "Confidence"]].copy()
submission_out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_out.shape)
print(submission_out.head())

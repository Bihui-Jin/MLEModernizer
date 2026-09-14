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

-6.856507064793506

# 6. Current score

-8.10766

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.63716) has done: 'I fix the dataset path detection so `TRAIN_CSV/TEST_CSV/SAMPLE_SUB_CSV` always resolve to the actual files in your environment, which unblocks the first failure and prevents all downstream `NameError`s. Then I keep your existing pipeline intact (same feature engineering, same CT model usage if weights exist, and the same linear fallback if not) but make it robust to missing image folders and ensure the final predictions align exactly to `sample_submission.csv` via a `Patient_Week` index mapping. Finally, I guarantee a valid `submission.csv` is written with the required columns and no missing rows, clipping confidence to the metric’s minimum as intended. These changes are execution/stability fixes and should move score from “no submission” to a valid baseline (and better if the pretrained weights are present).'
- What this solution (achieved -8.73145) has done: 'Your current score (-8.63716) is worse than the target (-6.8565), so we should improve (increase) it with minimal risk while preserving your core pipeline. The biggest safe lever for this competition is calibrating `Confidence` (sigma): your linear fallback currently uses per-patient residual std which can be too small/large and hurts Laplace log-likelihood; we compute a global MAE-based sigma (scaled for Laplace) and blend it with per-patient sigma, then clip to a reasonable range (>=70) to better match the metric. We also apply a small, safe shrinkage of the slope toward a robust global slope to reduce over-extrapolation on the final weeks (which usually reduces |Δ|), without changing the overall “per-patient linear” logic. The CT model path (if weights exist) is left intact; these changes mainly improve the fallback (and also help if some images are missing and you hit the zero-image path).'
- What this solution (achieved -8.73145) has done: 'We keep your pipeline intact and only tune two score-critical, low-risk levers in the fallback baseline that directly affect the Laplace log-likelihood: (1) slightly stronger slope shrinkage toward the robust global slope to reduce extrapolation error on the final weeks, and (2) more metric-aligned confidence calibration by using a blended per-patient/global Laplace sigma and clipping to a tighter upper bound to avoid over-penalizing via the log(sigma) term. We also ensure confidence is always finite and apply the same calibrated confidence at baseline week (still clipped at 70) to avoid inconsistencies. These changes should move the score upward (less negative) toward your target without changing the model architecture or training approach. The script still writes a valid `submission.csv` with the required columns and exact row alignment to `sample_submission.csv`.'
- What this solution (achieved -8.10766) has done: 'Your current score (-8.73145) is worse than the target (-6.8565), so we should improve (increase) it with minimal risk while keeping your per-patient linear fallback logic intact. The safest lever for this metric is calibrating `Confidence` (sigma): your current sigma can be miscalibrated and the metric strongly penalizes both under- and over-confidence; we fit a single global sigma from out-of-fold residuals and blend it with per-patient sigma more conservatively. We also make the slope shrinkage adaptive based on how many training points a patient has (less shrink when the patient trend is well-estimated; more shrink when it’s noisy), which typically reduces extrapolation error on the final weeks without changing the linear model core. Finally, we keep the pretrained CT model path untouched, and ensure the CSV alignment/writing remains exactly as required.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import scipy.ndimage
import pydicom

import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)




## === cell 1
def _find_osic_input_dir():
    """
    Fix: previous logic could return a directory that *contains* an extra nested
    'osic-pulmonary-fibrosis-progression' but then point to a non-existent train.csv path.
    We search common roots and accept only a directory that actually contains the needed CSVs.
    """
    candidates = [
        "/kaggle/input/osic-pulmonary-fibrosis-progression",
        "/kaggle/data/osic-pulmonary-fibrosis-progression",
        "../data/osic-pulmonary-fibrosis-progression",
        "../input/osic-pulmonary-fibrosis-progression",
        "/kaggle/input",
        "/kaggle/data",
        "../input",
        "../data",
    ]

    expanded = []
    for c in candidates:
        expanded.append(c)
        expanded.append(os.path.join(c, "osic-pulmonary-fibrosis-progression"))

    seen = set()
    for c in expanded:
        if c in seen:
            continue
        seen.add(c)
        if not os.path.isdir(c):
            continue
        if os.path.isfile(os.path.join(c, "train.csv")) and os.path.isfile(
            os.path.join(c, "sample_submission.csv")
        ):
            return c

    raise FileNotFoundError(
        "Could not locate osic-pulmonary-fibrosis-progression dataset directory with train.csv and sample_submission.csv."
    )


INPUT_DIR = _find_osic_input_dir()
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

print("Using INPUT_DIR:", INPUT_DIR)
print("TRAIN_CSV exists:", os.path.isfile(TRAIN_CSV))
print("TEST_CSV exists:", os.path.isfile(TEST_CSV))
print("SAMPLE_SUB_CSV exists:", os.path.isfile(SAMPLE_SUB_CSV))



## === cell 2
TRAIN_FOLDER = TRAIN_IMG_DIR


def load_scan(path):
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
        print("HU conversion Failed!!")
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
    path = dir_name + os.sep + patientid
    if not os.path.isdir(path):
        return None
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
    dir_name_of_patientid = TEST_IMG_DIR

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
            x_image = loaded_images.get(patientid[0], None)
            if x_image is None:
                x_image = np.zeros((100, 200, 200), dtype=np.float32)
            x_image = (
                torch.tensor(x_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            x_feature = x_features[i].unsqueeze(0)

            if torch.cuda.is_available() and device == "cuda":
                x_image = x_image.cuda()
                x_feature = x_feature.cuda()

            prediction = model(x_image, x_feature)
            predictions.append(prediction.to("cpu").detach().numpy()[0])

    predictions = np.array(predictions)
    npEval["FVC"] = predictions[:, 1]
    npEval["Confidence"] = predictions[:, 0]
    return npEval




## === cell 5
data_train = pd.read_csv(TRAIN_CSV)
data_test = pd.read_csv(TEST_CSV)
submission = pd.read_csv(SAMPLE_SUB_CSV)

print(
    "train:",
    data_train.shape,
    "test:",
    data_test.shape,
    "sample_sub:",
    submission.shape,
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
submission = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]].copy()
submission = submission.rename(columns={"base_FVC": "FVC"})



## === cell 7
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
npData = pd.concat([npData, data], axis=0, ignore_index=True, sort=True)
npData = npData.fillna(0)

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



## === cell 8
WEIGHTS_CANDIDATES = [
    "../input/tt-05-690/Epoch5_Score6.906373289247222_Acc0.9272943559861341.pth",
    "/kaggle/input/tt-05-690/Epoch5_Score6.906373289247222_Acc0.9272943559861341.pth",
]

weights_path = None
for p in WEIGHTS_CANDIDATES:
    if os.path.isfile(p):
        weights_path = p
        break

use_model = weights_path is not None
print("Using pretrained CT model weights:", use_model, "|", weights_path)




## === cell 9
def _clinical_linear_baseline(train_df, expanded_test_df):
    """
    Keep the same core per-patient linear extrapolation (polyfit slope/intercept from Weeks->FVC),
    but make two metric-relevant minimal adjustments to move score up toward the target:

    1) Confidence calibration using out-of-fold residuals:
       - The Laplace metric is sensitive to sigma; a global sigma estimated from
         OOF residuals is typically better calibrated than raw per-patient in-sample residuals.
       - We still blend with per-patient sigma to preserve your per-patient logic.

    2) Adaptive slope shrinkage:
       - When a patient has many training points, their slope is reliable (less shrink).
       - When they have few/noisy points, shrink more toward global_slope to reduce extrapolation.
    """
    train_df = train_df.copy()

    slopes = {}
    intercepts = {}
    resid_abs_insample = {}  # for per-patient scale estimate
    npoints = {}

    for pid, g in train_df.groupby("Patient"):
        g = g.sort_values("Weeks")
        x = g["Weeks"].values.astype(np.float64)
        y = g["FVC"].values.astype(np.float64)
        npoints[pid] = int(len(g))
        if len(g) >= 2 and np.std(x) > 0:
            b1, b0 = np.polyfit(x, y, 1)  # y = b1*x + b0
            yhat = b1 * x + b0
            resid = (y - yhat).astype(np.float64)
            slopes[pid] = float(b1)
            intercepts[pid] = float(b0)
            resid_abs_insample[pid] = np.abs(resid)
        else:
            slopes[pid] = 0.0
            intercepts[pid] = float(np.median(y)) if len(y) else 2000.0
            resid_abs_insample[pid] = np.array([200.0], dtype=np.float64)

    global_slope = float(np.median(list(slopes.values()))) if len(slopes) else 0.0

    oof_abs = []
    for pid, g in train_df.groupby("Patient"):
        g = g.sort_values("Weeks")
        x = g["Weeks"].values.astype(np.float64)
        y = g["FVC"].values.astype(np.float64)
        if len(g) < 3 or np.std(x) == 0:
            continue
        for i in range(len(g)):
            xm = np.delete(x, i)
            ym = np.delete(y, i)
            if len(xm) < 2 or np.std(xm) == 0:
                continue
            b1, b0 = np.polyfit(xm, ym, 1)
            pred = b1 * x[i] + b0
            oof_abs.append(abs(y[i] - pred))

    if len(oof_abs) == 0:
        global_mae = float(np.median(np.concatenate(list(resid_abs_insample.values()))))
    else:
        global_mae = float(np.median(np.array(oof_abs, dtype=np.float64)))
    global_sigma_lap = float(np.sqrt(2.0) * global_mae)

    out = expanded_test_df.copy()
    pid = out["Patient"].values
    base_fvc = out["base_FVC"].values.astype(np.float64)
    dt = out["Week"].values.astype(np.float64) - out["base_Weeks"].values.astype(
        np.float64
    )

    slope_vec_raw = np.array(
        [slopes.get(p, global_slope) for p in pid], dtype=np.float64
    )

    n_vec = np.array([npoints.get(p, 0) for p in pid], dtype=np.float64)
    shrink = 0.55 + 0.35 * (1.0 - np.exp(-np.clip(n_vec, 0, 20) / 5.0))
    slope_vec = shrink * slope_vec_raw + (1.0 - shrink) * global_slope

    sigma_lap_vec = []
    for p in pid:
        abs_res = resid_abs_insample.get(p, None)
        if abs_res is None or len(abs_res) == 0:
            mae_p = global_mae
        else:
            mae_p = float(np.median(abs_res))
        sigma_p = float(np.sqrt(2.0) * mae_p)
        sigma_blend = 0.50 * sigma_p + 0.50 * global_sigma_lap
        sigma_lap_vec.append(sigma_blend)

    sigma_vec = np.array(sigma_lap_vec, dtype=np.float64)

    out["FVC"] = base_fvc + slope_vec * dt

    out["Confidence"] = np.clip(
        np.nan_to_num(sigma_vec, nan=200.0, posinf=500.0, neginf=200.0), 70.0, 300.0
    )
    return out




## === cell 10
if use_model:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Combined_NET()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    test_pred = make_eval_data(data_test.copy(), model, device=device)
else:
    test_pred = _clinical_linear_baseline(data_train, data_test)



## === cell 11
test_pred["Confidence"] = np.nan_to_num(
    test_pred["Confidence"].astype(float).values, nan=200.0, posinf=500.0, neginf=200.0
)
test_pred.loc[test_pred.Confidence < 70, "Confidence"] = 70.0

for nid in test_pred.Patient.unique():
    idx = test_pred[
        (test_pred.Patient == nid) & (test_pred.Week == test_pred.base_Weeks)
    ].index.values
    if len(idx) > 0:
        test_pred.iloc[idx[0], test_pred.columns.get_loc("FVC")] = test_pred.iloc[
            idx[0], test_pred.columns.get_loc("base_FVC")
        ]
        test_pred.iloc[idx[0], test_pred.columns.get_loc("Confidence")] = 70.0



## === cell 12
pred_map = test_pred[["Patient", "Week", "FVC", "Confidence"]].copy()
pred_map["Patient_Week"] = (
    pred_map["Patient"].astype(str) + "_" + pred_map["Week"].astype(int).astype(str)
)
pred_map = pred_map.set_index("Patient_Week")[["FVC", "Confidence"]]

sub_out = pd.read_csv(SAMPLE_SUB_CSV).copy()
sub_out = sub_out.set_index("Patient_Week")

missing = sub_out.index.difference(pred_map.index)
if len(missing) > 0:
    base_map = pd.read_csv(TEST_CSV).set_index("Patient")["FVC"].to_dict()
    global_fvc = float(np.median(list(base_map.values()))) if len(base_map) else 2000.0
    fill_fvc = []
    fill_conf = []
    for pw in missing:
        pid = pw.split("_")[0]
        fill_fvc.append(float(base_map.get(pid, global_fvc)))
        fill_conf.append(70.0)
    pred_map_missing = pd.DataFrame(
        {"FVC": fill_fvc, "Confidence": fill_conf}, index=missing
    )
    pred_map = pd.concat([pred_map, pred_map_missing], axis=0)

sub_out[["FVC", "Confidence"]] = pred_map.loc[sub_out.index][
    ["FVC", "Confidence"]
].values
sub_out = sub_out.reset_index()

sub_out["FVC"] = sub_out["FVC"].astype(float)
sub_out["Confidence"] = sub_out["Confidence"].astype(float)
sub_out["Confidence"] = np.clip(
    np.nan_to_num(sub_out["Confidence"].values, nan=200.0, posinf=500.0, neginf=200.0),
    70.0,
    300.0,
)

sub_out.to_csv("submission.csv", index=False)
print(sub_out.head())
print("Wrote submission.csv with shape:", sub_out.shape)
print("Columns:", list(sub_out.columns))

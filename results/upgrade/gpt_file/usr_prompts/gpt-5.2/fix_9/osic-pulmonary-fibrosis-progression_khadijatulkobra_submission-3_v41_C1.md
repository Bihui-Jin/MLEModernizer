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

-6.894647832723742

# 6. Current score

-7.87805

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.61035) has done: 'I fix the immediate runtime breakages so the notebook runs end-to-end and writes a valid `submission.csv`. First, I replace the deprecated `DataFrame.append` call with `pd.concat` (pandas 2+ compatibility). Second, because the referenced pretrained `.pth` file is not available in this environment, I keep the same model/prediction pipeline but add a safe fallback that generates reasonable baseline predictions (use the provided baseline FVC and a calibrated constant confidence) so a submission is always produced. Finally, I correct the dataset paths to the provided `../data/osic-pulmonary-fibrosis-progression/...` location and make `make_eval_data` use that same path.'
- What this solution (achieved -8.33519) has done: 'We’re currently below the target (current -7.61035 vs target -6.89465; higher is better), so the smallest safe way to move upward is to improve the baseline fallback (used when pretrained weights are missing) without changing the model core. I keep your pipeline intact but replace the “predict base_FVC for all weeks” fallback with a per-patient linear trend estimated from train.csv (FVC vs Weeks), then apply it to each test week; this is a standard, lightweight improvement for this competition. I also calibrate a patient-level constant Confidence from the residual spread in training (clipped to Kaggle’s 70 floor), which typically improves the Laplace metric versus an arbitrary 250. Submission format/ordering stays exactly as your current code produces.'
- What this solution (achieved -7.58751) has done: 'You’re currently below the target (current score is more negative than target), so we should make the smallest legitimate improvement to the existing linear-fallback without touching the CNN pipeline. The main issue is that the fallback fits an intercept in absolute week space, which can be misaligned with the provided test baseline (`base_FVC` at `base_Weeks`); instead we anchor each patient’s prediction at their test baseline and only borrow the *slope* from train, which is both minimal and usually improves this metric. We also calibrate `Confidence` from the training residuals *after anchoring* and keep it constant per patient (clipped at 70), which typically improves the Laplace score versus an arbitrary or mismatched sigma. All paths, submission formatting, and model logic remain unchanged; only the fallback computation is adjusted.'
- What this solution (achieved -8.64155) has done: 'We’re below the target (current -7.58751 vs target -6.89465; higher is better), so we make the smallest safe improvement to the existing linear fallback without touching the CNN path. The main gain is to calibrate `Confidence` to the competition’s Laplace metric by estimating a robust global sigma from *one-step (slope-only, baseline-anchored)* residuals on the training set, then blending it with a per-patient sigma when available. This keeps your “anchor at test baseline + borrow slope from train” core logic identical, but makes sigma better matched to the actual error distribution, which typically improves the score. We also avoid forcing baseline-week confidence to exactly 70 (which can over-penalize if your errors are >70 elsewhere) and instead keep the calibrated sigma while still respecting the 70 floor.'
- What this solution (achieved -8.64155) has done: 'Your current score (-8.64155) is below the target (-6.89465), so we should make a small, legitimate improvement that doesn’t touch your CNN path and only adjusts the fallback (which is what actually runs when weights are missing). The main issue is that the fallback slope is estimated from raw patient FVC-vs-week, which is noisy; a minimal but effective fix is to estimate slope from **demeaned** data (per-patient mean-centered weeks/FVC), which stabilizes slope while keeping the same linear “anchor at baseline + slope*(week-base_week)” logic. Then we recalibrate Confidence consistently with that updated slope using a robust global + per-patient median absolute residual, still clipped at 70. Submission ordering/format stays identical; only the slope/sigma estimation in the fallback is adjusted to move the score upward toward the target.'
- What this solution (achieved -9.5147) has done: 'We’re below the target (current −8.64155 vs target −6.89465; higher is better), so we make a small improvement only in the fallback path (used when the pretrained `.pth` is missing) while leaving the CNN path untouched. The main change is to estimate each patient’s slope more robustly by fitting to all of their training points anchored at their **median week/FVC** (less sensitive to early outliers than anchoring at the first visit), while still predicting as `base_FVC + slope*(Week-base_Weeks)` for test. Then we recalibrate `Confidence` from the anchored residuals using a robust MAD-based estimate and a tiny blend with per-patient residual spread, which better matches the Laplace metric without changing evaluation semantics. Submission formatting/ordering and all file paths remain identical; the code still writes a valid `submission.csv`.'
- What this solution (achieved -8.73311) has done: 'We’re currently below the target (current −9.5147 vs target −6.8946; higher is better), and your pipeline is already relying on the fallback (no pretrained weights), so the smallest productive changes are in that fallback only. I keep the same “anchor at test baseline + per-patient linear slope” logic, but estimate slopes more robustly using a simple Theil–Sen-style median-of-pairwise-slopes (less sensitive to outliers than least-squares around a median anchor). Then I recalibrate `Confidence` consistently from the resulting residuals using a robust MAE→Laplace-b scale (and blend lightly with a global sigma), which better matches the competition metric while preserving evaluation semantics. Everything else (paths, model code, submission formatting) remains unchanged and it still write `submission.csv`.'
- What this solution (achieved -7.87805) has done: 'You’re below the target (current −8.733 vs target −6.895; higher is better), so we should make a small, legitimate improvement only in the fallback path (since pretrained weights are missing) while keeping the same “anchor at test baseline + linear slope” semantics. The current Theil–Sen slope uses only 16 evenly-spaced points when many visits exist, which can throw away useful information; we instead compute the median of *all* pairwise slopes for typical per-patient visit counts (small in this dataset) with a safe cap, improving slope quality without changing the model form. Next, we calibrate Confidence directly to the Laplace metric by estimating the Laplace scale `b = median(|resid|)/ln(2)` (instead of `sqrt(2)*MAE`) and then converting to sigma-like via `sigma = sqrt(2)*b`, which is a closer match to the competition’s likelihood and typically improves score. Everything else (data prep, CNN path, submission alignment/format) stays unchanged and it still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import pydicom  # this one is to read the dicom files
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import measure, morphology
from sklearn.preprocessing import normalize

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset



## === cell 1
DATA_ROOT = "../data/osic-pulmonary-fibrosis-progression"
TRAIN_FOLDER = os.path.join(DATA_ROOT, "train")
TEST_FOLDER = os.path.join(DATA_ROOT, "test")


def load_scan(path):  # Here path == (.../train/patientId)
    try:
        slices = [pydicom.dcmread(path + os.sep + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        files = os.listdir(path)
        files.sort()
        slices = [pydicom.dcmread(path + os.sep + s) for s in files]
    return slices


def make_dict_with_slices(
    patientIDs,
):  # this will be very huge memory consuming . don't use it
    patient_slices_dict = {}
    failed_slices_patiendIDs = []
    for patientID in patientIDs:
        path = TRAIN_FOLDER + os.sep + patientID
        try:
            patient_slices_dict[patientID] = load_scan(path)
        except Exception:
            failed_slices_patiendIDs.append(patientID)
    return patient_slices_dict, failed_slices_patiendIDs


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
        first_patient_pixels = get_pixels_hu(
            slices
        )  ## conversion to np array and HU unit
    else:
        first_patient_pixels = slices
    print("Number of Total Slices in this Scan:", len(slices))
    try:
        print("Shape of the the Image is:", slices.shape[1], slices.shape[2])
    except Exception:
        print("Shape of the the Image is:BLANK")
    fig = plt.figure(figsize=(10, 10))
    for i, slice_ in enumerate(first_patient_pixels[:16]):
        y = fig.add_subplot(4, 4, i + 1)
        y.imshow(slice_, cmap="gray")
    plt.show()


def resize_along_zaxis(slices, target_dimension=30):
    present_dimension = len(slices)
    if target_dimension == present_dimension:
        return slices
    zoom_factor = float(target_dimension) / float(present_dimension)
    resize_image = scipy.ndimage.zoom(slices, [zoom_factor, 1.0, 1.0])
    return resize_image


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


def save_array(
    patientID_folder, output_folder, Z=100, Y=200, X=200
):  # patientID_folder, output_folder
    patientIDs = os.listdir(patientID_folder)
    patientIDs.sort()
    Save_dir = output_folder  # './trainset'
    if not os.path.exists(Save_dir):
        print("The output directory doesnt exists")
        raise Exception

    for i, patientID in enumerate(patientIDs):
        path = TRAIN_FOLDER + os.sep + patientIDs[i]
        slices = load_scan(path)
        try:
            image_array = get_pixels_hu(slices)  # HU unit conversion + nparray
            ctimage_resizedAll = resize_along_allaxis(
                image_array,
                target_dimensionX=X,
                target_dimensionY=Y,
                target_dimensionZ=Z,
            )
            image = (image_normalize(ctimage_resizedAll) * 255.0).astype("uint8")
            np.save(Save_dir + os.sep + patientID + ".npy", image)
        except Exception:
            print("PatientId:%s couldnt be save and converted" % (patientID))


def load_array(path):  ### path='./traindataset/ID00052637202186188008618.npy'
    try:
        if path.endswith(".npy"):
            image_array = np.load(path)
        else:
            path = path + ".npy"
            image_array = np.load(path)
    except Exception:
        print(
            "The file in the Path:%s doesnetexists!!"
            % (path.split("\\")[-1].split("/")[-1].split(".")[0])
        )
        return []
    return image_array


def read_image(
    dir_name, patientid, Z=100, Y=200, X=200
):  # patientID_folder, output_folder
    path = dir_name + os.sep + patientid
    slices = load_scan(path)
    try:
        image_array = get_pixels_hu(slices)  # HU unit conversion + nparray
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
            npData = pd.concat([npData, data.loc[data.index == index[0]]], sort=False)
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
    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    c1_same_shape = torch.ones(sigma.size(), device=y_pred.device) * C1
    sigma_clip = torch.max(sigma, c1_same_shape)
    delta = (y_true[:, 0] - fvc_pred).abs()

    c2_same_shape = torch.ones(delta.size(), device=y_pred.device) * C2
    delta = torch.min(delta, c2_same_shape)

    sq2 = torch.tensor(2.0).sqrt()
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return (metric).mean()


def quartile_loss(y_true, y_pred):  # 0.65
    loss = score(y_true, y_pred)  # 0.35
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
    npEval["Confidence"] = predictions[:, 0]

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
submission = merge.loc[:, ["Patient_Week", "base_FVC", "Confidence"]]
submission = submission.rename(columns={"base_FVC": "FVC"})



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
npData = pd.concat([npData, data], axis=0, ignore_index=True, sort=True)
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
model = Combined_NET()
weights_path = (
    "../input/ww-19-674/Epoch19_Score6.741750120958745_Acc0.9344641051545048.pth"
)

use_model = False
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    use_model = True
else:
    use_model = False




## === cell 10
def _robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _theil_sen_slope(x: np.ndarray, y: np.ndarray, max_pairs: int = 4096) -> float:
    """
    Change (score-driven, minimal):
    Use more of each patient's available visits by taking the median of *all*
    pairwise slopes when feasible (typical n per patient is small here).
    This keeps the same linear slope-only fallback but yields a more reliable slope.
    A conservative cap keeps runtime bounded.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        return 0.0

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    slopes = []
    for i in range(n - 1):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n):
            dx = x[j] - xi
            if dx != 0:
                slopes.append((y[j] - yi) / dx)
                if len(slopes) >= max_pairs:
                    break
        if len(slopes) >= max_pairs:
            break

    if len(slopes) == 0:
        return 0.0
    return float(np.median(np.asarray(slopes, dtype=float)))


def _fit_patient_slope_and_sigma(train_df: pd.DataFrame):
    """
    Same fallback semantics:
      pred_FVC = base_FVC + slope*(Week - base_Weeks)

    Change (score-driven, minimal):
      - sigma calibration is aligned to the Laplace metric:
          For Laplace(0, b), median(|e|) = b * ln(2)
          => b = median(|e|) / ln(2)
          Kaggle metric uses sigma and sqrt(2); with sigma_clip=max(sigma,70)
          we keep using sigma-like value with:
              sigma = sqrt(2) * b
        This keeps confidence in the same units and typically improves the likelihood.
    """
    out = {}
    all_abs_resid = []
    all_slopes = []

    ln2 = float(np.log(2.0))
    sq2 = float(np.sqrt(2.0))

    for pid, g in train_df.groupby("Patient"):
        g = g.sort_values("Weeks")
        x = g["Weeks"].astype(float).values
        y = g["FVC"].astype(float).values

        if len(x) >= 2 and np.std(x) > 0:
            slope = _theil_sen_slope(x, y)
            x_anchor = float(np.median(x))
            y_anchor = float(np.median(y))
        else:
            slope = 0.0
            x_anchor = float(x[0]) if len(x) else 0.0
            y_anchor = float(y[0]) if len(y) else 0.0

        y_hat = y_anchor + slope * (x - x_anchor)
        resid = y - y_hat
        abs_resid = np.abs(resid) if len(resid) else np.array([250.0], dtype=float)

        med_abs = float(np.median(abs_resid)) if len(abs_resid) else 250.0
        b = med_abs / ln2 if ln2 > 0 else med_abs
        sigma = float(max(70.0, sq2 * b))

        out[pid] = (float(slope), float(sigma))
        all_abs_resid.extend(abs_resid.tolist())
        all_slopes.append(float(slope))

    if len(all_abs_resid) == 0:
        global_sigma = 250.0
    else:
        global_med_abs = float(np.median(np.asarray(all_abs_resid, dtype=float)))
        global_b = global_med_abs / ln2 if ln2 > 0 else global_med_abs
        global_sigma = float(max(70.0, sq2 * global_b))

    global_slope = float(np.median(all_slopes)) if len(all_slopes) else 0.0
    return out, global_slope, global_sigma


if use_model:
    test = make_eval_data(data_test.copy(), model)
else:
    patient_params, global_slope, global_sigma = _fit_patient_slope_and_sigma(
        data_train
    )
    test = data_test.copy()

    slopes = np.zeros(len(test), dtype=np.float64)
    sigmas = np.zeros(len(test), dtype=np.float64)
    for i, pid in enumerate(test["Patient"].values):
        if pid in patient_params:
            s, sg = patient_params[pid]
            sg = 0.7 * float(sg) + 0.3 * float(global_sigma)
        else:
            s, sg = float(global_slope), float(global_sigma)
        slopes[i] = float(s)
        sigmas[i] = float(sg)

    week = test["Week"].astype(float).values
    base_week = test["base_Weeks"].astype(float).values
    base_fvc = test["base_FVC"].astype(float).values
    test["FVC"] = (base_fvc + slopes * (week - base_week)).astype(float)
    test["Confidence"] = sigmas.astype(float)



## === cell 11
for nid in test.Patient.unique():
    index = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(index) > 0:
        test.iloc[index[0], test.columns.get_loc("FVC")] = test.iloc[
            index[0], test.columns.get_loc("base_FVC")
        ]



## === cell 12
test.loc[test.Confidence < 70, "Confidence"] = 70



## === cell 13
submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

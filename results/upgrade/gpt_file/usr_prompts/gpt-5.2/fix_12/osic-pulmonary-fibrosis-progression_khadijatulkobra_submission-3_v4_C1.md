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

-6.878090011498434

# 6. Current score

-7.61035

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.38565) has done: 'I fix the Pandas `DataFrame.append` deprecation by replacing it with `pd.concat`, which unblocks preprocessing and keeps identical semantics. Then I fix the missing model weights issue by loading an available `.pth` file from the Kaggle input tree if present; if none exists, the code fall back to a simple, deterministic baseline predictor (using `base_FVC` and a safe constant confidence) so a valid submission is always produced. I also correct hardcoded paths so they match your provided dataset location and make inference robust on CPU/GPU via `map_location`. Finally, I ensure the produced `submission.csv` has exactly the required columns and row alignment with `sample_submission.csv`.'
- What this solution (achieved -7.61035) has done: 'Your current score is worse than the target (gap ≈ -1.51), so we should improve accuracy/calibration but keep the same model and inference logic. The biggest safe win here is correcting the `Week` feature used by the model: it was overwritten to “relative week” and later incorrectly treated as “absolute week” when merging back to `sample_submission`, which can misalign predictions and hurt score. I preserve both `AbsWeek` (for submission alignment) and `Week` (relative, for the model feature) to ensure predictions map to the right `Patient_Week`. I also ensure `Confidence` is always positive and clipped to at least 70 (as required by the metric) to avoid pathological sigma values decreasing the score.'
- What this solution (achieved -7.61035) has done: 'Your current score (-7.61035) is worse than the target (-6.87809), so we should improve (increase) the metric with minimal, low-risk changes. The safest gain here is calibrating the predicted uncertainty: your model outputs three quantiles, but converting them to `Confidence` as `|q75-q25|` tends to be under-dispersed for the Laplace metric; we instead use a slightly larger, quantile-consistent sigma derived from the interquartile range. We also use the median quantile (`q50`) for `FVC` prediction (instead of the middle output by position, assuming it is `q50`, but we keep that as-is and just document it) and apply only a mild multiplicative calibration to sigma to better match the metric without changing the model. Finally, we ensure deterministic inference settings and keep the submission alignment exactly the same.'
- What this solution (achieved -7.61035) has done: 'We keep your model and preprocessing intact and only adjust the uncertainty calibration because your current gap to target comes primarily from the Laplace metric’s sensitivity to sigma. Specifically, we (1) compute `Confidence` from the predicted IQR using the correct Laplace relationship `b = IQR/(2 ln 2)` and `sigma = sqrt(2)*b`, then (2) apply a small, fixed calibration factor slightly larger than your current 1.15 to move the score upward toward the target without changing the predicted `FVC`. Finally, we add a conservative fallback so `Confidence` never becomes non-finite/too small even if q75≈q25 for some rows, which can otherwise harm the metric. These are minimal changes confined to prediction post-processing and should move the score closer to -6.878.'
- What this solution (achieved -7.61035) has done: 'We keep your model, preprocessing, and inference unchanged, and only adjust the post-processing that converts predicted quantiles into `Confidence`, because the Laplace metric is very sensitive to sigma calibration and your current gap to target suggests under/over-dispersion rather than FVC point error. Specifically, we tune the fixed multiplicative factor applied to `sigma_from_iqr` from 1.25 to a slightly larger value to improve the likelihood term while leaving `FVC=q50` intact. We also clip `Confidence` immediately in `make_eval_data` (not just later) to ensure the merged submission cannot contain too-small sigmas due to edge cases, without changing any model outputs. These are minimal, metric-aligned changes intended to move the score upward toward the target band.'
- What this solution (achieved -7.61035) has done: 'We keep your model and inference exactly the same and only adjust the uncertainty post-processing to better match the Laplace metric, since your current gap to target suggests sigma calibration is the main lever. Specifically, we replace the fixed `1.40` multiplier with a slightly larger but still mild factor so the likelihood term improves without changing the predicted `FVC=q50`. We also add a very small additive floor to the IQR-based sigma before clipping to reduce the chance of overly sharp (too-small) confidence when q75≈q25, which can heavily penalize the metric. All paths, merging/alignment to `sample_submission.csv`, and submission schema remain unchanged, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved -7.61035) has done: 'We keep your model, preprocessing, and inference intact and only adjust the `Confidence` calibration derived from the predicted IQR, because the Laplace log-likelihood is highly sensitive to sigma and your current gap to target suggests suboptimal dispersion rather than a major FVC point-error issue. Specifically, we slightly reduce the current sigma inflation (1.55×) to a milder factor so the likelihood term improves without changing `FVC=q50`. We keep the same IQR→sigma formula and the same safety floors/clips (including ≥70) to preserve evaluation semantics and avoid pathological small sigmas. All paths and the submission merge/alignment logic remain unchanged, and the script still always write a valid `submission.csv`.'
- What this solution (achieved -7.61035) has done: 'Your current score (-7.61035) is below the target (-6.87809), so we should improve (increase) the metric with the smallest, safest change. The most metric-aligned lever without touching the model or features is the post-processing that converts predicted quantiles into `Confidence`: we slightly reduce the current sigma inflation factor (1.40×) because over-dispersion can hurt the Laplace log-likelihood via the `-log(sigma)` term. We keep the same IQR→sigma formula, keep `FVC=q50` unchanged, and retain the same finite checks and `>=70` clipping so submission validity is preserved. Everything else (paths, image reading, model weights loading, merge/alignment, output schema) stays the same and still produces `submission.csv`.'
- What this solution (achieved -7.61035) has done: 'We keep your model, preprocessing, and inference unchanged, and only make a small, metric-aligned adjustment to the `Confidence` calibration derived from the predicted IQR. Since your current score (-7.61035) is below the target (-6.87809) (higher is better), we want a modest improvement; the lowest-risk lever is to slightly reduce the current sigma inflation factor so you’re less over-dispersed (improves the `-log(sigma)` term without touching `FVC=q50`). We also keep the existing finite checks and the mandatory `>=70` clipping to preserve valid Laplace scoring behavior. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -7.61035) has done: 'We keep your model, preprocessing, and inference identical, and only make a small metric-aligned adjustment to the `Confidence` calibration derived from the predicted IQR. Your current score (-7.61035) is below the target (-6.87809), so we want to increase the score; a common cause of overly negative Laplace scores is over-dispersed sigma, which hurts via the `-log(sigma)` term. We therefore slightly reduce the fixed sigma inflation multiplier (currently 1.15×) to a smaller value while keeping the same IQR→sigma formula, the same additive floor, and the same mandatory `>=70` clipping. This is a minimal, low-risk post-processing change that should move the score upward toward the target without changing predicted `FVC=q50`.'
- What this solution (achieved -7.61035) has done: 'Your current score (-7.61035) is worse than the target (-6.87809), so we should cautiously improve the Laplace log-likelihood without changing the model or features. The most sensitive, low-risk lever is `Confidence` (sigma): right now it’s derived from IQR but then inflated and floored in a way that may be slightly miscalibrated for this metric. I keep `FVC=q50` unchanged and only (1) use the Laplace-consistent IQR→sigma conversion without extra inflation, and (2) apply a very small additive floor to avoid rare near-zero IQR cases, while still clipping at >=70 as required. Everything else (paths, preprocessing, model loading, alignment to `sample_submission`, and writing `submission.csv`) remains the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import pydicom  # read DICOM
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

BASE_INPUT = "../input/osic-pulmonary-fibrosis-progression"
TRAIN_FOLDER = os.path.join(BASE_INPUT, "train")
TEST_FOLDER = os.path.join(BASE_INPUT, "test")

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




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


def make_dict_with_slices(patientIDs):  # huge memory, not used
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


def save_array(patientID_folder, output_folder, Z=100, Y=200, X=200):
    patientIDs = os.listdir(patientID_folder)
    patientIDs.sort()
    Save_dir = output_folder
    if not os.path.exists(Save_dir):
        print("The output directory doesnt exists")
        raise Exception
    for i, patientID in enumerate(patientIDs):
        path = TRAIN_FOLDER + os.sep + patientIDs[i]
        slices = load_scan(path)
        try:
            image_array = get_pixels_hu(slices)
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


def load_array(path):
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
            nn.Linear(41, 64), nn.ReLU(), nn.Linear(64, 119), nn.ReLU()
        )
        self.data_net2 = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 503), nn.ReLU()
        )
        self.data_net3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 119), nn.ReLU()
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

    c1_same_shape = torch.ones(sigma.size(), device=y_pred.device) * C1
    sigma_clip = torch.max(sigma, c1_same_shape)
    delta = (y_true[:, 0] - fvc_pred).abs()

    c2_same_shape = torch.ones(delta.size(), device=y_pred.device) * C2
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
def make_eval_data(npEval, model, device="cuda"):
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
    dir_name_of_patientid = TEST_FOLDER

    for unique_patient in unique_patients:
        loaded_images[unique_patient] = read_image(
            dir_name_of_patientid, unique_patient, Z=100, Y=200, X=200
        )

    use_cuda = torch.cuda.is_available() and device == "cuda"
    if use_cuda:
        model.to("cuda")
    model.eval()

    predictions = []
    for i, patientid in enumerate(x_patientids_name):
        x_image = loaded_images[patientid[0]]
        if x_image is None:
            x_image = np.zeros((100, 200, 200), dtype=np.uint8)

        x_image = torch.tensor(x_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x_feature = x_features[i].unsqueeze(0)

        if use_cuda:
            x_image = x_image.cuda(non_blocking=True)
            x_feature = x_feature.cuda(non_blocking=True)

        with torch.no_grad():
            prediction = model(x_image, x_feature)
        predictions.append(prediction.to("cpu").detach().numpy()[0])

    predictions = np.array(predictions)

    q25 = predictions[:, 0]
    q50 = predictions[:, 1]
    q75 = predictions[:, 2]

    iqr = np.abs(q75 - q25).astype(np.float64)

    sigma_from_iqr = iqr * (np.sqrt(2.0) / (2.0 * np.log(2.0)))

    sigma_cal = sigma_from_iqr + 20.0

    sigma_cal = np.where(np.isfinite(sigma_cal), sigma_cal, 250.0)
    sigma_cal = np.maximum(sigma_cal, 1.0)
    sigma_cal = np.maximum(sigma_cal, 70.0)

    npEval["FVC"] = q50
    npEval["Confidence"] = sigma_cal
    return npEval




## === cell 6
data_train = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
data_test_raw = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))
submission_template = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))



## === cell 7
submission_template["Patient"] = submission_template["Patient_Week"].apply(
    lambda x: x.split("_")[0]
)
submission_template["Weeks"] = (
    submission_template["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
submission_template = submission_template.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)

merge = (
    pd.merge(data_test_raw, submission_template, on=["Patient"], how="left")
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
submission = merge.loc[:, ["Patient_Week"]].copy()  # will fill FVC/Confidence later

del data_test_raw, submission_template, merge



## === cell 8
data = data_test.copy()
data["AbsWeek"] = data["Week"].astype(int)

data["Healthy-FVC"] = round((data["base_FVC"] * 100) / data["Percent"])
FE = ["Healthy-FVC"]

COLS = ["Sex", "SmokingStatus"]
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

FE1 = ["Male", "Female", "Ex-smoker", "Never smoked", "Currently smokes"]
npData = pd.DataFrame(
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"]
    + FE1
    + ["Week", "AbsWeek"]
)

npData = pd.concat([npData, data], ignore_index=True, sort=True)
npData = npData.fillna(0)

npData["Week"] = (
    npData["Week"] - npData["base_Weeks"]
)  # relative week feature for model
npData["base_Weeks"] = 0.0

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
        "Week",  # relative
        "AbsWeek",  # absolute (for merge back to submission)
    ]
].copy()
del npData




## === cell 9
def find_any_pth_file(search_root="../input"):
    candidates = []
    for root, dirs, files in os.walk(search_root):
        for fn in files:
            if fn.endswith(".pth"):
                candidates.append(os.path.join(root, fn))
    candidates.sort()
    return candidates[0] if candidates else None




## === cell 10
model = Combined_NET()

pth_path = os.path.join(
    "../input/test678", "Epoch6_Score6.783478332288338_Acc0.933478733806899.pth"
)
if not os.path.exists(pth_path):
    pth_path = find_any_pth_file("../input")

loaded_model = False
if pth_path is not None and os.path.exists(pth_path):
    try:
        state = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(state)
        loaded_model = True
        print(f"Loaded model weights from: {pth_path}")
    except Exception as e:
        print(f"Found weights at {pth_path} but failed to load: {e}")
        loaded_model = False
else:
    print("No .pth weights found under ../input; using baseline predictions.")



## === cell 11
if loaded_model:
    test_pred = make_eval_data(data_test.copy(), model, device="cuda")
    test_pred["Confidence"] = test_pred["Confidence"].astype(float).clip(lower=70)
else:
    test_pred = data_test.copy()
    test_pred["FVC"] = test_pred["base_FVC"].astype(float)
    test_pred["Confidence"] = 250.0  # safe constant; clipped at 70 in metric
    print("Baseline predictions generated.")



## === cell 12
sub = submission.copy()
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

pred_df = test_pred.copy()
pred_df["Weeks"] = pred_df["AbsWeek"].astype(int)
pred_df = pred_df[["Patient", "Weeks", "FVC", "Confidence"]]

sub = sub.merge(pred_df, on=["Patient", "Weeks"], how="left")

sub["FVC"] = sub["FVC"].fillna(0).astype(float)
sub["Confidence"] = sub["Confidence"].fillna(250.0).astype(float)
sub["Confidence"] = sub["Confidence"].clip(lower=70)

sub = sub[["Patient_Week", "FVC", "Confidence"]]



## === cell 13
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

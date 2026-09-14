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

-6.934037693446107

# 6. Current score

-9.88481

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.65417) has done: 'I fix the pandas `DataFrame.append` deprecation by replacing it with a safe `pd.concat`, which unblocks preprocessing. I also fix the missing pretrained model file by adding a fallback path search and, if no weights are found, run the (already-defined) model with random weights but still produce a valid submission. To ensure end-to-end execution, I make the test image directory path consistent with the dataset layout and add a CPU-safe `map_location` when loading weights. Finally, I guarantee the submission is written as `submission.csv` with the required columns and enforce the competition’s confidence clipping rule (≥70).'
- What this solution (achieved -24.09812) has done: 'Your current score is far below the target (gap ≈ -17.72, higher is better), and the biggest reason is that inference is likely using random weights or the wrong file, and also producing non-sensical Confidence values because the model outputs are unconstrained (ReLU but not calibrated). I (1) make weight loading robust to common checkpoint formats (`state_dict`, `model_state_dict`) so you actually use the intended pretrained weights when present, (2) enforce sane post-processing for predictions: keep FVC within a realistic range and set Confidence to a stable, competition-safe constant (70) to move the score upward toward the target without changing the model, and (3) fix the merge/sort alignment so submission rows match `sample_submission.csv` order exactly. These are minimal changes that preserve the existing architecture and inference logic while improving metric behavior and reducing catastrophic penalties. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -14.45863) has done: 'Your current score (-24.10) is far below the target (-6.93), so we should improve performance with the smallest changes that keep your model and preprocessing intact. The biggest likely issue is that the pretrained weights are not being found/loaded, so inference is effectively random; I make weight discovery stricter (only accept checkpoints that actually match `Combined_NET`) and add a safe fallback that still produces a valid submission. Next, to better match the competition metric without changing architecture, I post-process the model’s two outputs into (FVC, sigma) in a numerically safe way and compute Confidence from a patient-level residual estimate derived from the model’s own week-0 prediction error on the provided baseline row (then clip at 70). Finally, I keep the submission row order exactly identical to `sample_submission.csv` to avoid any accidental misalignment penalties.'
- What this solution (achieved -7.65751) has done: 'I fix the runtime error by correcting the tabular feature construction: the model’s SIGMA head expects exactly 42 input features (it concatenates 10 clinical features with a 32-dim image embedding), but the current code builds 74 features, causing the matmul shape mismatch. I minimally change `_build_features_42` to output 10 features (matching the original clinical inputs) and keep the rest of the pipeline and model unchanged. Then I rerun inference, keep the existing post-processing (FVC clipping + confidence clipping ≥70), and ensure a valid `submission.csv` is written in the exact `sample_submission.csv` row order.'
- What this solution (achieved -9.88481) has done: 'Your current score (-7.6575) is worse than the target (-6.9340), so we should improve it with minimal, low-risk changes that better match the Laplace log-likelihood metric without touching the model or preprocessing. The biggest lever here is Confidence calibration: your current patient-wise sigma estimate is likely too large (over-penalizing via the log term), so I recalibrate Confidence using out-of-fold residuals from a simple per-patient linear fit on train (already used) but without the extra +150 inflation, and then apply a single global scaling factor chosen from training to maximize the metric proxy. I also ensure the baseline week correction uses the closest available baseline row (since Week equality can miss due to type/merge issues), avoiding unintended NaNs and improving FVC alignment slightly. Everything else (architecture, feature construction, inference) stays the same, and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import pydicom  # to read dicom files
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

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
TRAIN_FOLDER = "../data/train"


def load_scan(
    path,
):  # Here path == (../input/osic-pulmonary-fibrosis-progression/train/patientId)
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

    sq2 = torch.tensor(2.0, device=y_pred.device).sqrt()
    metric = (delta / sigma_clip) * sq2 + (sigma_clip * sq2).log()
    return (metric).mean()


def quartile_loss(y_true, y_pred):  # 0.65
    loss = score(y_true, y_pred)  # 0.35
    return loss




## === cell 5
def _build_features_42(df: pd.DataFrame) -> torch.Tensor:
    cols10 = [
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
    missing = [c for c in cols10 if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for features: {missing}")

    x10 = df[cols10].astype(np.float32).values  # (N,10)
    if x10.shape[1] != 10:
        raise ValueError(f"Expected 10 base features, got {x10.shape[1]}")
    return torch.tensor(x10, dtype=torch.float32)


def make_eval_data(npEval, model, device="cuda"):
    x_features = _build_features_42(npEval)
    x_patientids_name = npEval[["Patient"]].values

    unique_patients = npEval.Patient.unique()
    loaded_images = {}

    candidate_dirs = [
        "../data/osic-pulmonary-fibrosis-progression/test",
        "../input/osic-pulmonary-fibrosis-progression/test",
        "../data/test",
        "../input/test",
    ]
    dir_name_of_patientid = None
    for d in candidate_dirs:
        if os.path.isdir(d):
            dir_name_of_patientid = d
            break
    if dir_name_of_patientid is None:
        raise FileNotFoundError(
            f"Could not find test DICOM directory. Tried: {candidate_dirs}"
        )

    for unique_patient in unique_patients:
        loaded_images[unique_patient] = read_image(
            dir_name_of_patientid, unique_patient, Z=100, Y=200, X=200
        )
        if loaded_images[unique_patient] is None:
            loaded_images[unique_patient] = np.zeros((100, 200, 200), dtype=np.uint8)

    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
    else:
        model.to("cpu")
    model.eval()

    predictions = []
    for i, patientid in enumerate(x_patientids_name):
        x_image = loaded_images[patientid[0]]
        x_image = (
            torch.tensor(x_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )  # channel + batch
        x_feature = x_features[i].unsqueeze(0)

        if torch.cuda.is_available() and device == "cuda":
            x_image = x_image.cuda(non_blocking=True)
            x_feature = x_feature.cuda(non_blocking=True)

        with torch.no_grad():
            prediction = model(x_image, x_feature)
        predictions.append(prediction.to("cpu").detach().numpy()[0])

    predictions = np.array(predictions, dtype=np.float32)

    sigma_raw = predictions[:, 0]
    fvc_raw = predictions[:, 1]

    npEval["FVC"] = fvc_raw
    npEval["Confidence"] = sigma_raw
    return npEval




## === cell 6
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


train_csv = _first_existing(
    [
        "../data/osic-pulmonary-fibrosis-progression/train.csv",
        "../input/osic-pulmonary-fibrosis-progression/train.csv",
        "../data/train.csv",
        "../input/train.csv",
    ]
)
test_csv = _first_existing(
    [
        "../data/osic-pulmonary-fibrosis-progression/test.csv",
        "../input/osic-pulmonary-fibrosis-progression/test.csv",
        "../data/test.csv",
        "../input/test.csv",
    ]
)
sample_sub_csv = _first_existing(
    [
        "../data/osic-pulmonary-fibrosis-progression/sample_submission.csv",
        "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv",
        "../data/sample_submission.csv",
        "../input/sample_submission.csv",
    ]
)

if train_csv is None or test_csv is None or sample_sub_csv is None:
    raise FileNotFoundError(
        f"Missing required CSVs. train={train_csv}, test={test_csv}, sample={sample_sub_csv}"
    )

data_train = pd.read_csv(train_csv)
data_test = pd.read_csv(test_csv)
sample_submission = pd.read_csv(sample_sub_csv)



## === cell 7
submission = sample_submission.copy()
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["Weeks"] = submission["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

merge = pd.merge(
    submission[["Patient_Week", "Patient", "Weeks"]],
    data_test,
    on="Patient",
    how="left",
    suffixes=("", "_base"),
)

merge = merge.rename(
    columns={"Weeks_base": "base_Weeks", "FVC": "base_FVC", "Weeks": "Week"}
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
        "Patient_Week",
    ],
].copy()

submission = merge.loc[:, ["Patient_Week"]].copy()
submission["FVC"] = 0.0
submission["Confidence"] = 70.0

del merge
del sample_submission



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
    columns=["Patient", "base_Weeks", "base_FVC", "Age", "Healthy-FVC"]
    + FE1
    + ["Week", "Patient_Week"]
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
        "Patient_Week",
    ]
].copy()
del npData




## === cell 9
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    return None


def _strip_module_prefix(state):
    if state is None:
        return None
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _state_looks_like_combined_net(state):
    if state is None or not isinstance(state, dict) or len(state) == 0:
        return False
    required_prefixes = [
        "image.feature_extractor",
        "image.classifier",
        "image.flat",
        "data.data_net1",
        "data.data_net4",
    ]
    keys = list(state.keys())
    return all(any(k.startswith(pref) for k in keys) for pref in required_prefixes)


def find_first_weight_file():
    candidates = [
        "../input/ww-58-669/Epoch58_Score6.699539787406163_Acc0.9367340273809749.pth",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    search_roots = ["../input", "../data", "../kaggle/input"]
    for root in search_roots:
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if not (fn.endswith(".pth") or fn.endswith(".pt")):
                        continue
                    p = os.path.join(dirpath, fn)
                    try:
                        obj = torch.load(p, map_location=torch.device("cpu"))
                        state = _strip_module_prefix(_extract_state_dict(obj))
                        if _state_looks_like_combined_net(state):
                            return p
                    except Exception:
                        continue
    return None


def load_weights_robust(model, weights_path, device):
    obj = torch.load(weights_path, map_location=torch.device(device))
    state = _strip_module_prefix(_extract_state_dict(obj))
    if state is None:
        raise ValueError(f"Unsupported checkpoint format at: {weights_path}")
    model.load_state_dict(state, strict=True)
    return model


weights_path = find_first_weight_file()
model = Combined_NET()

device = "cuda" if torch.cuda.is_available() else "cpu"
if weights_path is not None:
    print("Loading weights from:", weights_path)
    model = load_weights_robust(model, weights_path, device=device)
else:
    print(
        "WARNING: No compatible pretrained weights found; using randomly initialized model. Submission will be valid but score may be low."
    )



## === cell 10
train_proc = csv_preprocess(data_train.copy())

train_sigmas = {}
train_abs_resid = []
for pid, g in train_proc.groupby("Patient"):
    g = g.sort_values("Week")
    x = g["Week"].astype(np.float32).values
    y = g["actual_FVC"].astype(np.float32).values
    if len(x) >= 2 and np.std(x) > 0:
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - (a * x + b)
        s = float(np.std(resid))
        train_abs_resid.extend(np.abs(resid).tolist())
    else:
        s = 200.0
    train_sigmas[pid] = float(np.clip(s, 70.0, 1000.0))

train_abs_resid = np.array(train_abs_resid, dtype=np.float32)
if train_abs_resid.size == 0:
    train_abs_resid = np.array([200.0], dtype=np.float32)

global_sigma = float(np.clip(np.median(list(train_sigmas.values())), 70.0, 1000.0))

delta_ref = float(np.clip(np.median(train_abs_resid), 1.0, 1000.0))
sq2 = np.sqrt(2.0)


def metric_proxy(delta, sigma):
    sigma_c = max(float(sigma), 70.0)
    d = min(float(delta), 1000.0)
    return -(sq2 * d / sigma_c) - np.log(sq2 * sigma_c)


scales = np.array([0.60, 0.70, 0.80, 0.90, 1.00], dtype=np.float32)
proxy_scores = []
for sc in scales:
    proxy_scores.append(metric_proxy(delta_ref, global_sigma * float(sc)))
best_scale = float(scales[int(np.argmax(proxy_scores))])
print(
    "Calibrated confidence scale:",
    best_scale,
    "global_sigma:",
    global_sigma,
    "delta_ref:",
    delta_ref,
)

test = make_eval_data(
    data_test.copy(), model, device="cuda" if torch.cuda.is_available() else "cpu"
)

test["FVC"] = test["FVC"].astype(np.float32).clip(500.0, 6000.0)

tmp = test.loc[:, ["Patient", "Week", "base_Weeks", "FVC", "base_FVC"]].copy()
tmp["Week"] = tmp["Week"].astype(np.int32)
tmp["base_Weeks"] = tmp["base_Weeks"].astype(np.int32)
tmp["absdiff"] = (tmp["Week"] - tmp["base_Weeks"]).abs()
base_rows = (
    tmp.sort_values(["Patient", "absdiff"]).groupby("Patient", as_index=True).head(1)
)
base_rows = base_rows.set_index("Patient")
delta0 = (
    base_rows["base_FVC"].astype(np.float32) - base_rows["FVC"].astype(np.float32)
).to_dict()
test["FVC"] = (
    test["FVC"].astype(np.float32)
    + test["Patient"].map(delta0).fillna(0.0).astype(np.float32)
).astype(np.float32)

test["Confidence"] = (
    test["Patient"].map(train_sigmas).fillna(global_sigma).astype(np.float32)
    * best_scale
).clip(70.0, 1000.0)



## === cell 11
pred_df = test.loc[:, ["Patient_Week", "FVC", "Confidence"]].copy()

submission = submission.merge(
    pred_df, on="Patient_Week", how="left", suffixes=("", "_pred")
)

submission["FVC"] = submission["FVC_pred"].astype(np.float32).fillna(0.0)
submission["Confidence"] = (
    submission["Confidence_pred"].astype(np.float32).fillna(70.0).clip(lower=70.0)
)

submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

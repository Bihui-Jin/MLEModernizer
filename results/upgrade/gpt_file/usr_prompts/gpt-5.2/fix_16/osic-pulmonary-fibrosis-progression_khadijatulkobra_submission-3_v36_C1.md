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

-7.087391766990966

# 6. Current score

-17.94004

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.64853) has done: 'I fix the runtime error caused by the removed `DataFrame.append` by replacing it with `pd.concat` while keeping the same feature construction logic. I also fix all path issues by using the competition dataset path that exists in your environment (`../input/osic-pulmonary-fibrosis-progression/...`) and add a safe checkpoint-loading fallback (so the notebook still runs even if the external pretrained `.pth` isn’t available). To keep the core model and inference logic unchanged, I won’t alter the network, loss, or prediction loop; the only prediction-related change is ensuring `Confidence` is clipped to the competition minimum (70) and that a valid `submission.csv` with the required columns is always written.'
- What this solution (achieved -15.5967) has done: 'I remove the hard failure when no pretrained `.pth` checkpoint is present and instead fall back to training the provided model on `train.csv` clinical features only (keeping the same network and loss) so the notebook always runs end-to-end. I minimally adapt `make_eval_data` so it won’t crash if DICOM reading fails or images are missing, by substituting a zero-volume image with the correct shape while preserving the same forward call signature. I also add a lightweight training loop (CPU/GPU) that fits the model to predict `(sigma, FVC)` from the tabular features, with `sigma` supervised using residuals so the output is metric-aligned without changing the core architecture. Finally, I ensure `Confidence` is made non-negative, clipped to 70, and a valid `submission.csv` with the required columns is always written.'
- What this solution (achieved -16.57781) has done: 'I fix the shape mismatch causing the `mat1 and mat2 shapes cannot be multiplied` error by ensuring the tabular feature matrix fed into `SIGMA` is always exactly 10 features (so that `data_i(10) + image_o(32) = 42` as the model expects). This keeps the core model and loss intact and only corrects the feature construction to match the hard-coded network input size. I then make inference use the same 10-feature schema (instead of the padded 42-feature fallback that currently breaks), so `test` is defined and downstream cells run. Finally, I ensure `Confidence` is non-negative and clipped to the competition minimum (70) and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -16.68823) has done: 'Your current score is far below the target, so the safest way to move upward is to fix two metric-alignment issues without changing the model architecture or training loop structure: (1) the implemented “score” is the negative of Kaggle’s metric, so training is optimizing the wrong direction; and (2) your model outputs `ReLU`-bounded values, so `FVC` is forced non-negative and `Confidence` is forced large, both of which typically hurt this competition. I keep the same network and fallback training approach, but change the loss to directly maximize the Kaggle metric (equivalently minimize its negative) and post-process predictions to enforce `Confidence>=70` while softly calibrating sigma and clamping FVC to a plausible range derived from training data (a small, metric-consistent adjustment). These are minimal, directly score-relevant changes and should move the score substantially toward the target band while preserving the core logic and producing the same submission format.'
- What this solution (achieved -19.36426) has done: 'You’re currently far below the target (higher is better), so the smallest score-relevant way to move upward is to (1) make the fallback training target match the competition objective by providing both `FVC` and a per-row `sigma` target derived from each patient’s empirical residual scale, and (2) stop artificially shrinking `Confidence` at inference (the current `0.5 * Confidence` usually hurts the Laplace log-likelihood). This preserves your model architecture and training loop structure (same forward path, same loss function, same optimizer/epochs/batch size), but fixes the supervision mismatch where you were training against `y_true[:,0]` only. I also keep the required Kaggle constraint `Confidence >= 70` and keep the submission formatting/ordering identical.'
- What this solution (achieved -17.94004) has done: 'Your current score (-19.36) is far below the target (-7.09), so we should improve the metric while keeping your architecture and overall pipeline intact. The biggest score-relevant issue in your fallback training is a label/metric mismatch: `quartile_loss` expects `y_true[:,0]` to be the true FVC, but you currently pass a 1-column tensor and also store FVC/sigma in reversed order, which makes training optimize against the wrong target. I minimally fix the training target tensor to be shape `(N,2)` with `[dummy_sigma_for_metric, true_fvc]`, feed it directly to `quartile_loss`, and keep your auxiliary sigma regression term. I also switch the CT input in fallback training from all-zeros to a single precomputed “mean embedding” (computed once from a few real CTs) so the SIGMA head learns with a more realistic image feature distribution without changing the model structure or inference logic.'
- What this solution (achieved -17.94004) has done: 'Your current score (-17.94) is far below the target (-7.09), so we should improve the metric with the smallest changes that don’t alter your core model or training loop structure. The biggest score-relevant bug is that your training target tensor `y_true` is built as `[FVC, sigma]` but `kaggle_metric/quartile_loss` interpret `y_true[:,0]` as the true FVC; meanwhile your model outputs `[sigma, fvc]`, so the loss is currently comparing sigma vs fvc, which badly mis-trains the network. I fix this by swapping `y_true` to `[true_fvc, target_sigma]` and (to keep the auxiliary sigma term consistent) comparing `pred[:,0]` against `target_sigma`. Everything else (architecture, loss definition, optimizer, epochs, inference path, submission formatting) stays the same.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import pydicom  # to read dicom files
import scipy.ndimage
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tqdm.auto import tqdm

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TRAIN_FOLDER = "../input/osic-pulmonary-fibrosis-progression/train"


def load_scan(path):  # Here path == (.../train/patientId)
    files = os.listdir(path)
    files.sort()
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in files]
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        pass
    return slices


def make_dict_with_slices(patientIDs):  # huge memory consuming . don't use it
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
    for i, slice in enumerate(first_patient_pixels[:16]):
        y = fig.add_subplot(4, 4, i + 1)
        y.imshow(slice, cmap="gray")
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




## === cell 1
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

    out_rows = []
    for pid in data["Patient"].unique():
        sub = data.loc[data["Patient"] == pid]
        base_row = sub.iloc[[0]].copy()
        weeks = sub["base_Weeks"].to_numpy()
        fvcs = sub["base_FVC"].to_numpy()
        for k in range(len(weeks)):
            r = base_row.copy()
            r["Week"] = weeks[k]
            r["actual_FVC"] = fvcs[k]
            out_rows.append(r)

    npData = pd.concat(out_rows, axis=0, ignore_index=True, sort=False)
    npData = npData.fillna(0)
    npData = npData[
        ["Patient", "base_Weeks", "base_FVC", "Age"]
        + FE1
        + ["Week", "Healthy-FVC", "actual_FVC"]
    ]
    return npData




## === cell 2
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




## === cell 3
C1_F, C2_F = 70.0, 1000.0
_SQRT2 = None


def kaggle_metric(y_true, y_pred):
    global _SQRT2
    if _SQRT2 is None or _SQRT2.device != y_pred.device:
        _SQRT2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device))

    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = torch.clamp(sigma, min=C1_F)
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2_F)

    metric = -(_SQRT2 * delta / sigma_clip) - torch.log(_SQRT2 * sigma_clip)
    return metric.mean()


def quartile_loss(y_true, y_pred):
    return -kaggle_metric(y_true, y_pred)




## === cell 4
def make_eval_data(npEval, model, device="cuda", batch_size=64, img_batch_size=2):
    feat_cols = [
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
    x_features_np = npEval[feat_cols].values.astype(np.float32, copy=False)
    patient_ids = npEval["Patient"].values

    use_cuda = torch.cuda.is_available() and device == "cuda"
    dev = "cuda" if use_cuda else "cpu"
    model.to(dev)
    model.eval()

    dir_name_of_patientid = "../input/osic-pulmonary-fibrosis-progression/test"
    unique_patients, inv = np.unique(patient_ids, return_inverse=True)

    ct_np = np.empty((len(unique_patients), 1, 100, 200, 200), dtype=np.float32)
    for i, pid in enumerate(unique_patients):
        img = read_image(dir_name_of_patientid, pid, Z=100, Y=200, X=200)
        if img is None or (isinstance(img, list) and len(img) == 0):
            img = np.zeros((100, 200, 200), dtype=np.uint8)
        ct_np[i, 0] = img.astype(np.float32, copy=False)

    img_emb = np.empty((len(unique_patients), 32), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(unique_patients), img_batch_size):
            e = min(s + img_batch_size, len(unique_patients))
            t_img = torch.from_numpy(ct_np[s:e])
            if use_cuda:
                t_img = t_img.cuda(non_blocking=True)
            emb = model.image(t_img).detach().cpu().numpy()
            img_emb[s:e] = emb.astype(np.float32, copy=False)

    preds = np.empty((len(npEval), 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(npEval), batch_size):
            end = min(start + batch_size, len(npEval))
            idx = inv[start:end].astype(np.int64, copy=False)

            t_feat = torch.from_numpy(x_features_np[start:end])
            t_emb = torch.from_numpy(img_emb[idx])

            if use_cuda:
                t_feat = t_feat.cuda(non_blocking=True)
                t_emb = t_emb.cuda(non_blocking=True)

            out = model.data(t_feat, t_emb).detach().cpu().numpy()
            preds[start:end] = out

    npEval["FVC"] = preds[:, 1]
    npEval["Confidence"] = preds[:, 0]
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
del submission

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



## === cell 8
device = "cuda" if torch.cuda.is_available() else "cpu"




## === cell 9
def build_train_dataframe(data_train_raw: pd.DataFrame) -> pd.DataFrame:
    df = data_train_raw.copy()
    df["Healthy-FVC"] = round((df["FVC"] * 100) / df["Percent"])

    df["Male"] = (df["Sex"] == "Male").astype(int)
    df["Female"] = (df["Sex"] == "Female").astype(int)
    df["Ex-smoker"] = (df["SmokingStatus"] == "Ex-smoker").astype(int)
    df["Never smoked"] = (df["SmokingStatus"] == "Never smoked").astype(int)
    df["Currently smokes"] = (df["SmokingStatus"] == "Currently smokes").astype(int)

    df = df.sort_values(["Patient", "Weeks"]).reset_index(drop=True)
    df = df.rename(columns={"Weeks": "Week", "FVC": "actual_FVC"})
    df["base_Weeks"] = df.groupby("Patient")["Week"].transform("first")
    df["base_FVC"] = df.groupby("Patient")["actual_FVC"].transform("first")

    df = df[
        [
            "Patient",
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
            "actual_FVC",
        ]
    ].fillna(0)

    sigmas = np.empty(len(df), dtype=np.float32)
    for pid, g in df.groupby("Patient", sort=False):
        w = g["Week"].to_numpy(dtype=np.float32)
        y = g["actual_FVC"].to_numpy(dtype=np.float32)
        if len(g) >= 2:
            a, b = np.polyfit(w, y, deg=1)
            yhat = a * w + b
            resid = np.abs(y - yhat)
            s = float(np.median(resid))
        else:
            s = 200.0
        s = float(np.clip(s, 70.0, 1000.0))
        sigmas[g.index.values] = s

    df["target_sigma"] = sigmas
    return df


def make_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feat_cols = [
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
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return df[feat_cols].astype(np.float32).values


def compute_mean_image_embedding(
    model: Combined_NET, device: str, n_patients: int = 4
) -> np.ndarray:
    use_cuda = torch.cuda.is_available() and device == "cuda"
    dev = torch.device("cuda" if use_cuda else "cpu")

    train_dir = "../input/osic-pulmonary-fibrosis-progression/train"
    pids = []
    try:
        pids = sorted(os.listdir(train_dir))
    except Exception:
        pids = []
    pids = [p for p in pids if os.path.isdir(os.path.join(train_dir, p))][
        : max(1, int(n_patients))
    ]

    if len(pids) == 0:
        return np.zeros((32,), dtype=np.float32)

    model = model.to(dev)
    model.eval()

    embs = []
    with torch.no_grad():
        for pid in pids:
            img = read_image(train_dir, pid, Z=100, Y=200, X=200)
            if img is None or (isinstance(img, list) and len(img) == 0):
                continue
            t_img = torch.from_numpy(
                img.astype(np.float32, copy=False)[None, None, ...]
            )
            if use_cuda:
                t_img = t_img.cuda(non_blocking=True)
            e = model.image(t_img).detach().cpu().numpy()[0]
            embs.append(e.astype(np.float32, copy=False))

    if len(embs) == 0:
        return np.zeros((32,), dtype=np.float32)

    return np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32, copy=False)


def train_fallback_model(
    model: Combined_NET, train_df: pd.DataFrame, device: str
) -> Combined_NET:
    use_cuda = torch.cuda.is_available() and device == "cuda"
    dev = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(dev)

    x_train = make_feature_matrix(train_df)

    y_fvc = train_df["actual_FVC"].values.astype(np.float32, copy=False)
    y_sigma = train_df["target_sigma"].values.astype(np.float32, copy=False)

    y_true = np.zeros((len(train_df), 2), dtype=np.float32)
    y_true[:, 0] = y_fvc
    y_true[:, 1] = y_sigma

    train_dir = "../input/osic-pulmonary-fibrosis-progression/train"
    any_pid = None
    try:
        for p in sorted(os.listdir(train_dir)):
            if os.path.isdir(os.path.join(train_dir, p)):
                any_pid = p
                break
    except Exception:
        any_pid = None

    if any_pid is not None:
        img0 = read_image(train_dir, any_pid, Z=100, Y=200, X=200)
        if img0 is None or (isinstance(img0, list) and len(img0) == 0):
            img0 = np.zeros((100, 200, 200), dtype=np.uint8)
        img0 = img0.astype(np.float32, copy=False)
    else:
        img0 = np.zeros((100, 200, 200), dtype=np.float32)

    x_img = np.repeat(img0[None, None, ...], repeats=len(train_df), axis=0).astype(
        np.float32, copy=False
    )

    ds = TensorDataset(
        torch.from_numpy(x_img),
        torch.from_numpy(x_train),
        torch.from_numpy(y_true),
    )
    loader = DataLoader(
        ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=use_cuda
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    epochs = 2
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"fallback-train ep {ep+1}/{epochs}", leave=False)
        for b_img, b_x, b_ytrue in pbar:
            b_img = b_img.to(dev, non_blocking=use_cuda)
            b_x = b_x.to(dev, non_blocking=use_cuda)
            b_ytrue = b_ytrue.to(dev, non_blocking=use_cuda)

            pred = model(b_img, b_x)  # (sigma, fvc)

            loss_main = quartile_loss(b_ytrue, pred)

            loss_sigma = F.l1_loss(pred[:, 0], b_ytrue[:, 1])

            loss = loss_main + 0.1 * loss_sigma

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model


model = Combined_NET()
ckpt_path = "../input/ww-02-680/Epoch2_Score6.80294730836982_Acc0.9312787661489272.pth"

if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    print("Loaded checkpoint:", ckpt_path)
else:
    print(
        "Checkpoint not found, running fallback training on clinical features:",
        ckpt_path,
    )
    train_df = build_train_dataframe(data_train)
    model = train_fallback_model(model, train_df, device=device)



## === cell 10
test = make_eval_data(
    data_test.copy(), model, device=device, batch_size=256, img_batch_size=2
)



## === cell 11
for nid in test.Patient.unique():
    idx = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(idx) > 0:
        test.iloc[idx[0], test.columns.get_loc("FVC")] = test.iloc[
            idx[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[idx[0], test.columns.get_loc("Confidence")] = 70



## === cell 12
fvc_min = float(np.percentile(data_train["FVC"].values, 0.5))
fvc_max = float(np.percentile(data_train["FVC"].values, 99.5))

test["Confidence"] = np.abs(test["Confidence"].values).astype(np.float32)
test.loc[test.Confidence < 70, "Confidence"] = 70

test["FVC"] = test["FVC"].astype(np.float32).clip(fvc_min, fvc_max)

submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values
submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

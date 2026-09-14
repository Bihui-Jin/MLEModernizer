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

-7.087391766990966

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.64853) has done: 'I fix the runtime error caused by the removed `DataFrame.append` by replacing it with `pd.concat` while keeping the same feature construction logic. I also fix all path issues by using the competition dataset path that exists in your environment (`../input/osic-pulmonary-fibrosis-progression/...`) and add a safe checkpoint-loading fallback (so the notebook still runs even if the external pretrained `.pth` isn’t available). To keep the core model and inference logic unchanged, I won’t alter the network, loss, or prediction loop; the only prediction-related change is ensuring `Confidence` is clipped to the competition minimum (70) and that a valid `submission.csv` with the required columns is always written.'
- What this solution (achieved -15.5967) has done: 'I remove the hard failure when no pretrained `.pth` checkpoint is present and instead fall back to training the provided model on `train.csv` clinical features only (keeping the same network and loss) so the notebook always runs end-to-end. I minimally adapt `make_eval_data` so it won’t crash if DICOM reading fails or images are missing, by substituting a zero-volume image with the correct shape while preserving the same forward call signature. I also add a lightweight training loop (CPU/GPU) that fits the model to predict `(sigma, FVC)` from the tabular features, with `sigma` supervised using residuals so the output is metric-aligned without changing the core architecture. Finally, I ensure `Confidence` is made non-negative, clipped to 70, and a valid `submission.csv` with the required columns is always written.'

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
        first_patient_pixels = get_pixels_hu(
            slices
        )  # conversion to np array and HU unit
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


def score(y_true, y_pred):
    global _SQRT2
    if _SQRT2 is None or _SQRT2.device != y_pred.device:
        _SQRT2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device))

    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = torch.clamp(sigma, min=C1_F)
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2_F)

    metric = (delta / sigma_clip) * _SQRT2 + (sigma_clip * _SQRT2).log()
    return metric.mean()


def quartile_loss(y_true, y_pred):
    loss = score(y_true, y_pred)
    return loss




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
    return df




## === cell 10
def make_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    base_cols = ["base_Weeks", "base_FVC", "Age", "Week", "Healthy-FVC"]

    for c in base_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column for features: {c}")

    x_parts = [df[base_cols].astype(np.float32).values]

    for col in ["Sex", "SmokingStatus"]:
        if col in df.columns:
            uniques = pd.Series(df[col].fillna("")).unique().tolist()
            for u in uniques:
                name = str(u)
                if name == "":
                    continue
                x_parts.append(
                    (df[col].astype(str) == name).astype(np.float32).values[:, None]
                )

    x_raw = np.concatenate(x_parts, axis=1).astype(np.float32, copy=False)

    x42 = np.zeros((len(df), 42), dtype=np.float32)
    n = min(42, x_raw.shape[1])
    x42[:, :n] = x_raw[:, :n]
    return x42


def train_fallback_model(
    model: Combined_NET, train_df: pd.DataFrame, device: str
) -> Combined_NET:
    use_cuda = torch.cuda.is_available() and device == "cuda"
    dev = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(dev)

    x_train = make_feature_matrix(train_df)

    y_fvc = train_df["actual_FVC"].values.astype(np.float32, copy=False)
    y = np.zeros((len(train_df), 1), dtype=np.float32)
    y[:, 0] = y_fvc

    x_img = np.zeros((len(train_df), 1, 100, 200, 200), dtype=np.float32)

    ds = TensorDataset(
        torch.from_numpy(x_img),
        torch.from_numpy(x_train),
        torch.from_numpy(y),
    )
    loader = DataLoader(
        ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=use_cuda
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    epochs = 2
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"fallback-train ep {ep+1}/{epochs}", leave=False)
        for b_img, b_x, b_y in pbar:
            b_img = b_img.to(dev, non_blocking=use_cuda)
            b_x = b_x.to(dev, non_blocking=use_cuda)
            b_y = b_y.to(dev, non_blocking=use_cuda)

            pred = model(b_img, b_x)  # (sigma, fvc)

            loss_main = quartile_loss(b_y, pred)

            with torch.no_grad():
                target_sigma = (
                    (b_y[:, 0] - pred[:, 1]).abs().clamp(min=70.0, max=1000.0)
                )
            loss_sigma = F.l1_loss(pred[:, 0], target_sigma)

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




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1834299482.py in <cell line: 0>()
    104     )
    105     train_df = build_train_dataframe(data_train)
--> 106     model = train_fallback_model(model, train_df, device=device)
    107 
    108 

/tmp/ipykernel_55/1834299482.py in train_fallback_model(model, train_df, device)
     71             b_y = b_y.to(dev, non_blocking=use_cuda)
     72 
---> 73             pred = model(b_img, b_x)  # (sigma, fvc)
     74 
     75             loss_main = quartile_loss(b_y, pred)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/1840887106.py in forward(self, image_i, data_i)
    155     def forward(self, image_i, data_i):
    156         image_o = self.image(image_i)
--> 157         data_o = self.data(data_i, image_o)
    158         return data_o
    159 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/1840887106.py in forward(self, data_i, image_o)
     45     def forward(self, data_i, image_o):
     46         x = torch.cat((data_i, image_o), dim=-1)
---> 47         out1 = self.data_net1(x)
     48         out2 = torch.cat((data_i, out1), dim=-1)
     49         out2 = self.data_net2(out2)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x74 and 42x64)

## === cell 11
def make_eval_data_padded(
    npEval, model, device="cuda", batch_size=64, img_batch_size=2
):
    x_features_np = make_feature_matrix(npEval)
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


_raw_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")[
    ["Patient", "Sex", "SmokingStatus"]
]
data_test_with_cat = data_test.merge(_raw_test, on="Patient", how="left")

test = make_eval_data_padded(
    data_test_with_cat.copy(), model, device=device, batch_size=256, img_batch_size=2
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2710037517.py in <cell line: 0>()
     59 data_test_with_cat = data_test.merge(_raw_test, on="Patient", how="left")
     60 
---> 61 test = make_eval_data_padded(
     62     data_test_with_cat.copy(), model, device=device, batch_size=256, img_batch_size=2
     63 )

/tmp/ipykernel_55/2710037517.py in make_eval_data_padded(npEval, model, device, batch_size, img_batch_size)
     43                 t_emb = t_emb.cuda(non_blocking=True)
     44 
---> 45             out = model.data(t_feat, t_emb).detach().cpu().numpy()
     46             preds[start:end] = out
     47 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/1840887106.py in forward(self, data_i, image_o)
     45     def forward(self, data_i, image_o):
     46         x = torch.cat((data_i, image_o), dim=-1)
---> 47         out1 = self.data_net1(x)
     48         out2 = torch.cat((data_i, out1), dim=-1)
     49         out2 = self.data_net2(out2)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x74 and 42x64)

## === cell 12
for nid in test.Patient.unique():
    idx = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
    if len(idx) > 0:
        test.iloc[idx[0], test.columns.get_loc("FVC")] = test.iloc[
            idx[0], test.columns.get_loc("base_FVC")
        ]
        test.iloc[idx[0], test.columns.get_loc("Confidence")] = 70



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2160495900.py in <cell line: 0>()
----> 1 for nid in test.Patient.unique():
      2     idx = test[(test.Patient == nid) & (test.Week == test.base_Weeks)].index.values
      3     if len(idx) > 0:
      4         test.iloc[idx[0], test.columns.get_loc("FVC")] = test.iloc[
      5             idx[0], test.columns.get_loc("base_FVC")

NameError: name 'test' is not defined

## === cell 13
test["Confidence"] = np.abs(test["Confidence"].values)
test.loc[test.Confidence < 70, "Confidence"] = 70

submission.loc[:, "FVC"] = test.FVC.values
submission.loc[:, "Confidence"] = test.Confidence.values
submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1555339843.py in <cell line: 0>()
      1 # Keep metric-compliant Confidence
----> 2 test["Confidence"] = np.abs(test["Confidence"].values)
      3 test.loc[test.Confidence < 70, "Confidence"] = 70
      4 
      5 submission.loc[:, "FVC"] = test.FVC.values

NameError: name 'test' is not defined

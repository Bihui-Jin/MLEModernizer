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

-10.58042

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -24.64853) has done: 'The timeout is dominated by repeatedly building large image batches by expanding the same 3D CT tensor for every set of weeks per patient, which multiplies memory traffic and CPU compute. I keep the exact same model and inference semantics, but change `make_eval_data` to run one forward pass per patient by computing the image embedding once (`model.image(vol)`) and then running only the MLP head (`model.data(...)`) for all that patient’s rows in feature batches. I also avoid extra copies (use `np.asarray(..., dtype=np.float32, order="C")` and reuse tensors), and keep the existing on-disk caching of preprocessed volumes. These changes are mathematically equivalent to the original forward calls (same layers, same weights, same outputs) while cutting redundant compute drastically.'
- What this solution (achieved -24.64853) has done: 'I fix the broken merge/sort logic in the submission-week expansion so `base_Weeks/base_FVC` are created correctly (the current code sorts/refs a non-existent `Weeks` column and then tries to read `base_FVC` before it exists). Then I ensure the engineered feature columns expected by `make_eval_data()` are always present in `data_test` (including one-hot categories that may be missing in test) so inference doesn’t KeyError. Finally, I keep the model and inference core unchanged, but make sure the final `submission.csv` is written with the exact required columns and row alignment to the sample submission.'
- What this solution (achieved -10.58042) has done: 'Your current score is far below the target, so the safest way to move it upward is to fix prediction calibration rather than change the model. The biggest issue is that the network outputs are unconstrained and you are using them raw as `Confidence` and `FVC`, which can yield invalid/overconfident sigmas and wildly shifted FVC; the OSIC metric heavily penalizes that. I keep the exact same model and inference flow, but (1) convert the predicted sigma to a strictly-positive value using a monotonic transform and then clip to a reasonable range, and (2) anchor the predicted FVC to the baseline by converting the model’s FVC output into a delta added to `base_FVC`, then clip to a plausible physiological range. These are minimal post-processing steps consistent with the evaluation semantics (predict FVC + uncertainty) and typically produce a large score gain from very poor baselines.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from tqdm.auto import tqdm



## === cell 1
TRAIN_FOLDER = "../input/osic-pulmonary-fibrosis-progression/train"


def _sorted_dicom_paths(path):
    entries = os.listdir(path)
    entries.sort()
    return [os.path.join(path, s) for s in entries]


_DICOM_TAGS = [
    "PixelData",
    "RescaleIntercept",
    "RescaleSlope",
    "ImagePositionPatient",
]


def load_scan(path):
    files = _sorted_dicom_paths(path)
    slices = [pydicom.dcmread(fp, specific_tags=_DICOM_TAGS) for fp in files]
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        pass
    return slices


def make_dict_with_slices(patientIDs):
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
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16, copy=False)

    try:
        image[image <= -2000] = 0
        slopes = np.array(
            [float(getattr(s, "RescaleSlope", 1.0)) for s in slices], dtype=np.float32
        )
        intercepts = np.array(
            [float(getattr(s, "RescaleIntercept", 0.0)) for s in slices],
            dtype=np.float32,
        )

        if not np.all(slopes == 1.0):
            image = (image.astype(np.float32) * slopes[:, None, None]).astype(np.int16)
        image = (
            image.astype(np.int32) + intercepts[:, None, None].astype(np.int32)
        ).astype(np.int16)
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


def save_array(patientID_folder, output_folder, Z=100, Y=200, X=200):
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

    out_parts = []
    for pid, g in data.groupby("Patient", sort=False):
        base_row = g.iloc[[0]].copy()
        k = len(g)
        rep = pd.concat([base_row] * k, ignore_index=True)
        rep["Week"] = g["base_Weeks"].to_numpy()
        rep["actual_FVC"] = g["base_FVC"].to_numpy()
        out_parts.append(rep)

    npData = pd.concat(out_parts, ignore_index=True, sort=False)
    npData = npData.fillna(0)
    return npData




## === cell 3
class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


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
C1, C2 = 70.0, 1000.0
SQ2 = float(np.sqrt(2.0))


def score(y_true, y_pred):
    sigma = y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = torch.clamp(sigma, min=C1)
    delta = (y_true[:, 0] - fvc_pred).abs()
    delta = torch.clamp(delta, max=C2)

    metric = (delta / sigma_clip) * SQ2 + (sigma_clip * SQ2).log()
    return metric.mean()


def quartile_loss(y_true, y_pred):
    loss = score(y_true, y_pred)
    return loss




## === cell 5
def _cache_path(cache_dir, pid, Z, Y, X):
    return os.path.join(cache_dir, f"{pid}_Z{int(Z)}_Y{int(Y)}_X{int(X)}_uint8.npy")


def _load_or_build_vol_uint8(pid, img_root, cache_dir, Z=100, Y=200, X=200):
    os.makedirs(cache_dir, exist_ok=True)
    cp = _cache_path(cache_dir, pid, Z, Y, X)
    if os.path.exists(cp):
        return np.load(cp, mmap_mode="r")
    im = read_image(img_root, pid, Z=Z, Y=Y, X=X)
    if im is None:
        im = np.zeros((int(Z), int(Y), int(X)), dtype=np.uint8)
    np.save(cp, im)
    return np.load(cp, mmap_mode="r")


def make_eval_data(npEval, model, device="cpu", batch_size=64):
    feature_cols = [
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

    use_cuda = torch.cuda.is_available() and device == "cuda"
    dev = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(dev)
    model.eval()

    img_root = "../input/osic-pulmonary-fibrosis-progression/test"
    cache_dir = "./_osic_cache_test_volumes"
    Z, Y, X = 100, 200, 200

    df = npEval.reset_index(drop=True)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    feats_np = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    n = len(df)
    preds = np.empty((n, 2), dtype=np.float32)

    pid_groups = df.groupby("Patient", sort=False).indices

    with torch.inference_mode():
        for pid, idxs in pid_groups.items():
            vol = _load_or_build_vol_uint8(pid, img_root, cache_dir, Z=Z, Y=Y, X=X)
            vol_f32 = np.asarray(vol, dtype=np.float32, order="C")
            vol_t = torch.from_numpy(vol_f32).unsqueeze(0).unsqueeze(0).to(dev)

            image_embed = model.image(vol_t)  # (1, 32)

            idxs_arr = np.asarray(idxs, dtype=np.int64)
            f_all = torch.from_numpy(feats_np[idxs_arr]).to(dev)
            m = f_all.shape[0]

            out_parts = []
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                bs = end - start
                img_rep = image_embed.expand(bs, -1)
                out = model.data(f_all[start:end], img_rep)
                out_parts.append(out.cpu())
            out_full = torch.cat(out_parts, dim=0).numpy()
            preds[idxs_arr] = out_full

    out_df = df.copy()
    out_df["FVC"] = preds[:, 1]
    out_df["Confidence"] = preds[:, 0]
    return out_df




## === cell 6
data_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
data_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 7
pw = submission["Patient_Week"].str.split("_", n=1, expand=True)
submission["Patient"] = pw[0]
submission["Weeks"] = pw[1].astype(np.int32)

submission = submission.sort_values(
    by=["Patient", "Weeks"], ascending=True
).reset_index(drop=True)

merge = pd.merge(
    submission[["Patient_Week", "Patient", "Weeks"]],  # requested weeks
    data_test,  # baseline info incl. Weeks/FVC at baseline
    on="Patient",
    how="left",
)

merge = merge.sort_values(["Patient", "Weeks_x"], ascending=True).reset_index(drop=True)

data_test = (
    merge.loc[
        :,
        [
            "Patient",
            "Weeks_y",  # baseline week from test.csv
            "FVC",  # baseline fvc from test.csv
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks_x",  # target week to predict (from sample_submission)
        ],
    ]
    .rename(
        columns={
            "Weeks_y": "base_Weeks",
            "FVC": "base_FVC",
            "Weeks_x": "Week",
        }
    )
    .reset_index(drop=True)
)

submission = merge.loc[:, ["Patient_Week"]].copy()
submission["FVC"] = 0.0
submission["Confidence"] = 100.0

del merge
del data_train  # unused



## === cell 8
data = data_test.copy()

den = data["Percent"].replace(0, np.nan)
data["Healthy-FVC"] = np.round((data["base_FVC"] * 100.0) / den).fillna(0.0)

data["Male"] = (data["Sex"] == "Male").astype(np.float32)
data["Female"] = (data["Sex"] == "Female").astype(np.float32)

data["Ex-smoker"] = (data["SmokingStatus"] == "Ex-smoker").astype(np.float32)
data["Never smoked"] = (data["SmokingStatus"] == "Never smoked").astype(np.float32)
data["Currently smokes"] = (data["SmokingStatus"] == "Currently smokes").astype(
    np.float32
)

data = data.fillna(0.0)

data_test = data[
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
    ]
].reset_index(drop=True)
del data



## === cell 9
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except Exception:
    pass



## === cell 10
model = Combined_NET()

ckpt_path = (
    "../input/ww-19-674/Epoch19_Score6.741750120958745_Acc0.9344641051545048.pth"
)
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
else:
    print(
        f"WARNING: checkpoint not found at {ckpt_path}. Using randomly initialized weights (submission will be valid but score may be poor)."
    )



## === cell 11
test = make_eval_data(data_test.copy(), model, device="cpu", batch_size=64)



## === cell 12
raw_sigma = test["Confidence"].to_numpy(dtype=np.float32)
raw_fvc = test["FVC"].to_numpy(dtype=np.float32)
base_fvc = test["base_FVC"].to_numpy(dtype=np.float32)

sigma_pos = np.log1p(np.exp(raw_sigma)).astype(np.float32)

sigma_clip = np.clip(sigma_pos, 70.0, 400.0).astype(np.float32)

fvc_pred = (base_fvc + raw_fvc).astype(np.float32)

fvc_pred = np.clip(fvc_pred, 500.0, 6000.0).astype(np.float32)

test.loc[:, "FVC"] = fvc_pred
test.loc[:, "Confidence"] = sigma_clip

mask = test["Week"].to_numpy() == test["base_Weeks"].to_numpy()
test.loc[mask, "FVC"] = test.loc[mask, "base_FVC"].to_numpy()
test.loc[mask, "Confidence"] = 70.0



## === cell 13
if len(test) != len(submission):
    raise RuntimeError(
        f"Row mismatch: test preds={len(test)} vs submission template={len(submission)}"
    )

submission.loc[:, "FVC"] = test["FVC"].to_numpy()
submission.loc[:, "Confidence"] = test["Confidence"].to_numpy()

submission_out = submission[["Patient_Week", "FVC", "Confidence"]].copy()
submission_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_out.shape)
print(submission_out.head())
